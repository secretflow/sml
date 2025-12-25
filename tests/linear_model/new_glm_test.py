# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for GLM comparing with statsmodels.

NOTE: When comparing with statsmodels, we use y_scale=1.0 to disable
the MPC-oriented scaling that would otherwise shift the intercept.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sml.linear_model.glm.core.distribution import (
    Bernoulli,
    Gamma,
    InverseGaussian,
    NegativeBinomial,
    Normal,
    Poisson,
    Tweedie,
)
from sml.linear_model.glm.core.link import (
    CLogLogLink,
    IdentityLink,
    LogitLink,
    LogLink,
    PowerLink,
    ProbitLink,
    ReciprocalLink,
)
from sml.linear_model.glm.model import GLM

# Try to import statsmodels for comparison
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import family as sm_family
    from statsmodels.genmod.families import links as sm_links

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ==================== Data Generation Functions ====================


def generate_poisson_data(seed=42, n_samples=200, n_features=5):
    """Generate Poisson distributed data."""
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (n_samples, n_features))
    true_coef = jnp.array([0.5, -0.3, 0.2, 0.1, -0.1][:n_features])
    true_intercept = 1.0

    eta = X @ true_coef + true_intercept
    mu = jnp.exp(eta)
    y = jax.random.poisson(k2, mu).astype(jnp.float32)

    return X, y, true_coef, true_intercept


def generate_normal_data(seed=42, n_samples=100, n_features=3):
    """Generate Normal distributed data."""
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (n_samples, n_features))
    true_coef = jnp.array([1.0, -0.5, 0.3][:n_features])
    true_intercept = 2.0

    y = X @ true_coef + true_intercept + 0.1 * jax.random.normal(k2, (n_samples,))

    return X, y, true_coef, true_intercept


def generate_bernoulli_data(seed=42, n_samples=200, n_features=4):
    """Generate Bernoulli distributed data."""
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (n_samples, n_features))
    true_coef = jnp.array([1.0, -0.5, 0.5, -0.3][:n_features])
    true_intercept = 0.5

    eta = X @ true_coef + true_intercept
    prob = jax.nn.sigmoid(eta)
    y = jax.random.bernoulli(k2, prob).astype(jnp.float32)

    return X, y, true_coef, true_intercept


def generate_gamma_data(seed=42, n_samples=200, n_features=3, shape=5.0):
    """Generate Gamma distributed data using jax.random.gamma."""
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (n_samples, n_features)) * 0.5
    true_coef = jnp.array([0.5, -0.3, 0.2][:n_features])
    true_intercept = 1.5

    eta = X @ true_coef + true_intercept
    mu = jnp.exp(eta)

    # Gamma: mean = shape * scale, so scale = mu / shape
    y = jax.random.gamma(k2, shape, (n_samples,)) * (mu / shape)

    return X, y, true_coef, true_intercept


def generate_inverse_gaussian_data(seed=42, n_samples=200, n_features=3, lam=10.0):
    """Generate Inverse Gaussian (Wald) distributed data using numpy."""
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    X = jax.random.normal(key, (n_samples, n_features)) * 0.3
    true_coef = jnp.array([0.3, -0.2, 0.1][:n_features])
    true_intercept = 1.0

    eta = X @ true_coef + true_intercept
    mu = jnp.exp(eta)

    # Use numpy for proper wald distribution with lambda parameter
    y = jnp.array(np.random.wald(np.array(mu), lam))

    return X, y, true_coef, true_intercept


def generate_negative_binomial_data(seed=42, n_samples=200, n_features=3, alpha=1.0):
    """Generate Negative Binomial distributed data (Corrected)."""
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)

    X = jax.random.normal(k1, (n_samples, n_features))
    true_coef = jnp.array([0.5, -0.3, 0.2][:n_features])
    true_intercept = 1.0

    eta = X @ true_coef + true_intercept
    mu = jnp.exp(eta)

    # Correct Logic: Poisson-Gamma mixture
    # We need a Gamma variable with mean = mu and variance = alpha * mu^2
    # Gamma params: shape(k) = 1/alpha, scale(theta) = alpha * mu
    # JAX gamma uses shape parameter 'a', returns samples with scale=1.
    # So we multiply by the target scale.

    r = 1.0 / alpha
    scale = alpha * mu  # or mu / r

    # Generate mixing lambda
    gamma_samples = jax.random.gamma(k2, r, (n_samples,)) * scale

    # Generate y
    y = jax.random.poisson(k3, gamma_samples).astype(jnp.float32)

    return X, y, true_coef, true_intercept


# ==================== Helper Functions ====================


def compute_auc(y_true, y_prob):
    """Compute AUC-ROC for binary classification."""
    sorted_indices = jnp.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]

    n_pos = jnp.sum(y_true)
    n_neg = len(y_true) - n_pos

    tp_cumsum = jnp.cumsum(y_true_sorted)
    fp_cumsum = jnp.cumsum(1 - y_true_sorted)

    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg

    return jnp.trapezoid(tpr, fpr)


def compare_with_statsmodels(
    our_model,
    sm_model,
    X,
    y,
    coef_rtol=0.1,
    coef_atol=0.05,
    dev_rtol=0.1,
    is_binary=False,
):
    """
    Compare our GLM with statsmodels: coefficients and deviance.

    Returns a dict with comparison results.
    """
    # Extract statsmodels params
    sm_coef = sm_model.params[1:]
    sm_intercept = sm_model.params[0]
    sm_dev = sm_model.deviance

    # Our model predictions and deviance
    our_mu = our_model.predict(X)

    if is_binary:
        # For binary, compare AUC instead of deviance
        X_sm = sm.add_constant(np.array(X))
        sm_mu = sm_model.predict(X_sm)

        our_auc = compute_auc(y, our_mu)
        sm_auc = compute_auc(y, jnp.array(sm_mu))

        print(f"  AUC - Ours: {our_auc:.4f}, statsmodels: {sm_auc:.4f}")
        np.testing.assert_allclose(our_auc, sm_auc, rtol=0.05)
    else:
        our_dev = float(our_model.dist.deviance(y, our_mu))
        print(f"  Deviance - Ours: {our_dev:.4f}, statsmodels: {sm_dev:.4f}")
        np.testing.assert_allclose(our_dev, sm_dev, rtol=dev_rtol)

    # Compare coefficients
    print(f"  Coef - Ours: {our_model.coef_}")
    print(f"  Coef - SM:   {sm_coef}")
    print(f"  Intercept - Ours: {our_model.intercept_:.4f}, SM: {sm_intercept:.4f}")

    np.testing.assert_allclose(our_model.coef_, sm_coef, rtol=coef_rtol, atol=coef_atol)
    np.testing.assert_allclose(
        our_model.intercept_, sm_intercept, rtol=coef_rtol, atol=coef_atol
    )


# ==================== Test Class ====================


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
class TestGLMvsStatsmodelsPlaintext:
    """Test GLM by comparing with statsmodels."""

    # ==================== IRLS Tests ====================

    def test_poisson_irls(self):
        """Test Poisson GLM with IRLS solver vs statsmodels."""
        X, y, _, _ = generate_poisson_data()

        our_model = GLM(
            dist=Poisson(), solver="irls", fit_intercept=True, max_iter=50, tol=1e-8
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Poisson()).fit()

        print("Poisson IRLS:")
        compare_with_statsmodels(our_model, sm_model, X, y)

    def test_normal_irls(self):
        """Test Normal GLM with IRLS solver vs statsmodels."""
        X, y, _, _ = generate_normal_data()

        our_model = GLM(dist=Normal(), solver="irls", fit_intercept=True, max_iter=20)
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Gaussian()).fit()

        print("Normal IRLS:")
        compare_with_statsmodels(our_model, sm_model, X, y)

    def test_bernoulli_irls(self):
        """Test Bernoulli GLM with IRLS solver vs statsmodels."""
        X, y, _, _ = generate_bernoulli_data()

        our_model = GLM(
            dist=Bernoulli(),
            link=LogitLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Binomial()).fit()

        print("Bernoulli IRLS:")
        compare_with_statsmodels(our_model, sm_model, X, y, is_binary=True)

    def test_gamma_irls(self):
        """Test Gamma GLM with IRLS solver vs statsmodels."""
        X, y, _, _ = generate_gamma_data()

        our_model = GLM(
            dist=Gamma(),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.Log())
        ).fit()

        print("Gamma IRLS:")
        compare_with_statsmodels(our_model, sm_model, X, y)

    def test_inverse_gaussian_irls(self):
        """Test Inverse Gaussian GLM with IRLS solver vs statsmodels."""
        X, y, _, _ = generate_inverse_gaussian_data()

        our_model = GLM(
            dist=InverseGaussian(),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.InverseGaussian(link=sm_links.Log())
        ).fit()

        print("InverseGaussian IRLS:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.15, dev_rtol=0.15
        )

    def test_negative_binomial_irls(self):
        """Test Negative Binomial GLM with IRLS solver vs statsmodels."""
        X, y, _, _ = generate_negative_binomial_data(alpha=1.0)

        our_model = GLM(
            dist=NegativeBinomial(alpha=1.0),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.NegativeBinomial(alpha=1.0)
        ).fit()

        print("NegativeBinomial IRLS:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.15, dev_rtol=0.15
        )

    # ==================== SGD Tests ====================

    def test_poisson_sgd(self):
        """Test Poisson GLM with SGD solver vs statsmodels."""
        X, y, _, _ = generate_poisson_data()

        our_model = GLM(
            dist=Poisson(),
            solver="sgd",
            fit_intercept=True,
            max_iter=500,
            learning_rate=0.01,
            batch_size=32,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Poisson()).fit()

        print("Poisson SGD:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.15, coef_atol=0.1
        )

    def test_normal_sgd(self):
        """Test Normal GLM with SGD solver vs statsmodels."""
        X, y, _, _ = generate_normal_data()

        our_model = GLM(
            dist=Normal(),
            solver="sgd",
            fit_intercept=True,
            max_iter=1000,
            learning_rate=0.05,
            batch_size=32,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Gaussian()).fit()

        print("Normal SGD:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.2, coef_atol=0.15, dev_rtol=0.5
        )

    def test_bernoulli_sgd(self):
        """Test Bernoulli GLM with SGD solver vs statsmodels."""
        X, y, _, _ = generate_bernoulli_data()

        our_model = GLM(
            dist=Bernoulli(),
            link=LogitLink(),
            solver="sgd",
            fit_intercept=True,
            max_iter=500,
            learning_rate=0.01,
            batch_size=32,
        )
        our_model.fit(X, y, enable_spu_cache=False)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Binomial()).fit()

        print("Bernoulli SGD:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.2, coef_atol=0.15, is_binary=True
        )

    def test_gamma_sgd(self):
        """Test Gamma GLM with SGD solver vs statsmodels."""
        X, y, _, _ = generate_gamma_data(shape=10.0)

        our_model = GLM(
            dist=Gamma(),
            link=LogLink(),
            solver="sgd",
            fit_intercept=True,
            max_iter=500,
            learning_rate=0.01,
            batch_size=32,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.Log())
        ).fit()

        print("Gamma SGD:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.2, coef_atol=0.15, dev_rtol=0.15
        )

    @pytest.mark.skip(reason="InverseGaussian SGD is numerically challenging")
    def test_inverse_gaussian_sgd(self):
        """Test Inverse Gaussian GLM with SGD solver vs statsmodels."""
        X, y, _, _ = generate_inverse_gaussian_data(lam=50.0)

        our_model = GLM(
            dist=InverseGaussian(),
            link=LogLink(),
            solver="sgd",
            fit_intercept=True,
            max_iter=1000,
            learning_rate=1e-4,
            batch_size=64,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.InverseGaussian(link=sm_links.Log())
        ).fit()

        print("InverseGaussian SGD:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.25, coef_atol=0.2, dev_rtol=0.5
        )

    def test_negative_binomial_sgd(self):
        """Test Negative Binomial GLM with SGD solver vs statsmodels."""
        X, y, _, _ = generate_negative_binomial_data(alpha=1.0)

        our_model = GLM(
            dist=NegativeBinomial(alpha=1.0),
            solver="sgd",
            fit_intercept=True,
            max_iter=500,
            learning_rate=0.01,
            batch_size=32,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.NegativeBinomial(alpha=1.0)
        ).fit()

        print("NegativeBinomial SGD:")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.2, coef_atol=0.15, dev_rtol=0.2
        )

    # ==================== Optimized Solver Tests ====================

    def test_gamma_log_optimized_irls(self):
        """Test optimized Gamma+Log IRLS solver vs statsmodels."""
        from sml.linear_model.glm.solvers.irls import GammaLogIRLSSolver

        X, y, _, _ = generate_gamma_data()

        # Use the optimized solver directly
        solver = GammaLogIRLSSolver()
        from sml.linear_model.glm.core.family import Family

        family = Family(Gamma(), LogLink())

        # Fit using the optimized solver
        beta, _, history = solver.solve(
            X,
            y,
            family,
            fit_intercept=True,
            max_iter=50,
            tol=1e-8,
        )

        # Extract coef and intercept
        our_coef = beta[:-1]
        our_intercept = beta[-1]

        # Compute predictions and deviance
        X_with_intercept = jnp.hstack([X, jnp.ones((X.shape[0], 1))])
        eta = jnp.matmul(X_with_intercept, beta)
        mu = jnp.exp(eta)
        our_dev = float(Gamma().deviance(y, mu))

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.Log())
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]
        sm_dev = sm_model.deviance

        print("Gamma+Log Optimized IRLS:")
        print(f"  Deviance - Ours: {our_dev:.4f}, statsmodels: {sm_dev:.4f}")
        print(f"  Coef - Ours: {our_coef}")
        print(f"  Coef - SM:   {sm_coef}")
        print(f"  Intercept - Ours: {our_intercept:.4f}, SM: {sm_intercept:.4f}")

        np.testing.assert_allclose(our_dev, sm_dev, rtol=0.1)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=0.1, atol=0.05)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=0.1, atol=0.05)

    def test_tweedie_log_optimized_irls(self):
        """Test optimized Tweedie+Log IRLS solver vs statsmodels."""
        from sml.linear_model.glm.solvers.irls import TweedieLogIRLSSolver

        # Generate Tweedie-like data (using Gamma as approximation)
        X, y, _, _ = generate_gamma_data(shape=10.0)

        # Use the optimized solver directly
        power = 1.5
        solver = TweedieLogIRLSSolver(power=power)
        from sml.linear_model.glm.core.family import Family

        family = Family(Tweedie(power=power), LogLink())

        # Fit using the optimized solver
        beta, _, history = solver.solve(
            X,
            y,
            family,
            fit_intercept=True,
            max_iter=50,
            tol=1e-8,
        )

        # Extract coef and intercept
        our_coef = beta[:-1]
        our_intercept = beta[-1]

        # Compute predictions and deviance
        X_with_intercept = jnp.hstack([X, jnp.ones((X.shape[0], 1))])
        eta = jnp.matmul(X_with_intercept, beta)
        mu = jnp.exp(eta)
        our_dev = float(Tweedie(power=power).deviance(y, mu))

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y),
            X_sm,
            family=sm_family.Tweedie(var_power=power, link=sm_links.Log()),
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]
        sm_dev = sm_model.deviance

        print("Tweedie+Log Optimized IRLS:")
        print(f"  Deviance - Ours: {our_dev:.4f}, statsmodels: {sm_dev:.4f}")
        print(f"  Coef - Ours: {our_coef}")
        print(f"  Coef - SM:   {sm_coef}")
        print(f"  Intercept - Ours: {our_intercept:.4f}, SM: {sm_intercept:.4f}")

        np.testing.assert_allclose(our_dev, sm_dev, rtol=0.1)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=0.05, atol=0.01)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=0.05, atol=0.01)

    def test_optimized_solver_registry(self):
        """Test that optimized solvers are properly registered."""
        from sml.linear_model.glm.solvers.optimized_irls import (
            get_registered_solver,
            list_registered_solvers,
        )
        from sml.linear_model.glm.core.family import Family

        # Check Gamma+Log is registered
        gamma_log_family = Family(Gamma(), LogLink())
        solver = get_registered_solver(gamma_log_family)
        assert solver is not None, "Gamma+Log solver should be registered"

        # Check Tweedie+Log is registered
        tweedie_log_family = Family(Tweedie(power=1.5), LogLink())
        solver = get_registered_solver(tweedie_log_family)
        assert solver is not None, "Tweedie+Log(p=1.5) solver should be registered"

        # List all registered solvers
        registered = list_registered_solvers()
        print(f"Registered optimized solvers: {registered}")
        assert len(registered) >= 2


# ==================== Additional Feature Tests ====================


class TestGLMFeatures:
    """Test additional GLM features without statsmodels dependency."""

    def test_various_links_bernoulli(self):
        """Test Bernoulli with different link functions."""
        X, y, _, _ = generate_bernoulli_data()

        for link_cls, name in [
            (LogitLink, "logit"),
            (ProbitLink, "probit"),
        ]:
            model = GLM(
                dist=Bernoulli(),
                link=link_cls(),
                solver="irls",
                fit_intercept=True,
            )
            model.fit(X, y, enable_spu_cache=False)

            mu_pred = model.predict(X)
            assert jnp.all(mu_pred >= 0) and jnp.all(
                mu_pred <= 1
            ), f"{name} link failed"
            print(f"Bernoulli with {name}: OK")

    def test_power_link(self):
        """Test Power link function."""
        X, y, _, _ = generate_gamma_data()

        model = GLM(
            dist=Gamma(),
            link=PowerLink(power=0.5),  # Square root link
            solver="irls",
            fit_intercept=True,
            max_iter=30,
        )
        model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        mu_pred = model.predict(X)
        assert jnp.all(mu_pred > 0), "Gamma predictions should be positive"

    def test_reciprocal_link(self):
        """Test Reciprocal link function."""
        X, y, _, _ = generate_gamma_data()

        model = GLM(
            dist=Gamma(),
            link=ReciprocalLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=30,
        )
        model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        mu_pred = model.predict(X)
        assert jnp.all(mu_pred > 0), "Gamma predictions should be positive"

    def test_l2_regularization(self):
        """Test L2 regularization effect."""
        X, y, _, _ = generate_poisson_data()

        # Fit without regularization
        model_noreg = GLM(dist=Poisson(), solver="irls", fit_intercept=True, l2=0.0)
        model_noreg.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit with regularization
        model_reg = GLM(dist=Poisson(), solver="irls", fit_intercept=True, l2=1.0)
        model_reg.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Regularized coefficients should be smaller in magnitude
        assert jnp.sum(model_reg.coef_**2) < jnp.sum(model_noreg.coef_**2)

    def test_sample_weights(self):
        """Test sample weights handling."""
        X, y, _, _ = generate_poisson_data(n_samples=100)
        weights = jnp.ones(100)
        weights = weights.at[:50].set(2.0)

        model = GLM(dist=Poisson(), solver="irls", fit_intercept=True)
        model.fit(X, y, sample_weight=weights, enable_spu_cache=False, y_scale=1.0)

        assert model.coef_ is not None

    def test_offset_handling(self):
        """Test offset handling in prediction."""
        X, y, _, _ = generate_poisson_data(n_samples=100)
        offset = jnp.ones(100) * 0.5

        model = GLM(dist=Poisson(), solver="irls", fit_intercept=True)
        model.fit(X, y, offset=offset, enable_spu_cache=False, y_scale=1.0)

        # Predictions with offset should differ from without
        mu_with_offset = model.predict(X, offset=offset)
        mu_without_offset = model.predict(X)

        assert not jnp.allclose(mu_with_offset, mu_without_offset)

    def test_sgd_batching(self):
        """Test SGD with different batch sizes."""
        X, y, _, _ = generate_poisson_data()

        for batch_size in [16, 32, 64]:
            model = GLM(
                dist=Poisson(),
                solver="sgd",
                fit_intercept=True,
                max_iter=100,
                batch_size=batch_size,
            )
            model.fit(X, y, enable_spu_cache=False, y_scale=1.0)
            assert model.coef_ is not None
            print(f"SGD batch_size={batch_size}: OK")

    def test_convergence_history(self):
        """Test that convergence history is recorded."""
        X, y, _, _ = generate_poisson_data()

        model = GLM(dist=Poisson(), solver="irls", fit_intercept=True, max_iter=20)
        model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Check convergence history exists
        assert hasattr(model, "history_")

    def test_no_intercept(self):
        """Test GLM without intercept."""
        X, y, _, _ = generate_poisson_data()

        model = GLM(
            dist=Poisson(),
            solver="irls",
            fit_intercept=False,
        )
        model.fit(X, y, enable_spu_cache=False, y_scale=1.0)
        assert model.coef_ is not None
        assert model.intercept_ == 0.0 or model.intercept_ is None
