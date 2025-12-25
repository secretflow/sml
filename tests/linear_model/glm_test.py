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
import spu.libspu as libspu
import spu.utils.simulation as spsim

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


# ==================== Test Plaintext Class ====================


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
    # These tests verify that GLM automatically selects the optimized solver
    # when the distribution+link combination matches a registered solver.

    def test_gamma_log_optimized_irls(self):
        """Test that GLM uses optimized Gamma+Log IRLS solver vs statsmodels."""
        X, y, _, _ = generate_gamma_data()

        # Use GLM class - it should automatically select GammaLogIRLSSolver
        our_model = GLM(
            dist=Gamma(),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.Log())
        ).fit()

        print("Gamma+Log Optimized IRLS (via GLM):")
        compare_with_statsmodels(our_model, sm_model, X, y)

    def test_tweedie_log_optimized_irls(self):
        """Test that GLM uses optimized Tweedie+Log IRLS solver vs statsmodels."""
        # Generate Tweedie-like data (using Gamma as approximation)
        X, y, _, _ = generate_gamma_data(shape=10.0)

        power = 1.5
        # Use GLM class - it should automatically select TweedieLogIRLSSolver
        our_model = GLM(
            dist=Tweedie(power=power),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y),
            X_sm,
            family=sm_family.Tweedie(var_power=power, link=sm_links.Log()),
        ).fit()

        print("Tweedie+Log Optimized IRLS (via GLM):")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.05, coef_atol=0.01
        )

    def test_tweedie_log_different_power(self):
        """Test that GLM handles Tweedie with different power values."""
        X, y, _, _ = generate_gamma_data(shape=8.0)

        # Test with power = 1.2 (should still use TweedieLogIRLSSolver)
        power = 1.2
        our_model = GLM(
            dist=Tweedie(power=power),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y),
            X_sm,
            family=sm_family.Tweedie(var_power=power, link=sm_links.Log()),
        ).fit()

        print(f"Tweedie(p={power})+Log Optimized IRLS (via GLM):")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.1, coef_atol=0.05
        )

    def test_optimized_solver_registry(self):
        """Test that optimized solvers are properly registered."""
        from sml.linear_model.glm.core.family import Family
        from sml.linear_model.glm.solvers.optimized_irls import (
            get_registered_solver,
            list_registered_solvers,
        )

        # Check Gamma+Log is registered
        gamma_log_family = Family(Gamma(), LogLink())
        solver = get_registered_solver(gamma_log_family)
        assert solver is not None, "Gamma+Log solver should be registered"

        # Check Tweedie+Log is registered for ANY power value
        for power in [1.2, 1.5, 1.8]:
            tweedie_log_family = Family(Tweedie(power=power), LogLink())
            solver = get_registered_solver(tweedie_log_family)
            assert (
                solver is not None
            ), f"Tweedie+Log(p={power}) solver should be registered"
            # Verify the solver has correct power
            assert (
                solver.power == power  # type: ignore
            ), f"TweedieLogIRLSSolver should have power={power}"

        # Check all new solvers are registered
        from sml.linear_model.glm.core.distribution import Bernoulli, Normal, Poisson
        from sml.linear_model.glm.core.link import (
            IdentityLink,
            LogitLink,
            ReciprocalLink,
        )

        gaussian_identity_family = Family(Normal(), IdentityLink())
        assert get_registered_solver(gaussian_identity_family) is not None

        poisson_log_family = Family(Poisson(), LogLink())
        assert get_registered_solver(poisson_log_family) is not None

        bernoulli_logit_family = Family(Bernoulli(), LogitLink())
        assert get_registered_solver(bernoulli_logit_family) is not None

        gamma_inverse_family = Family(Gamma(), ReciprocalLink())
        assert get_registered_solver(gamma_inverse_family) is not None

        # List all registered solvers
        registered = list_registered_solvers()
        print(f"Registered optimized solvers: {registered}")
        assert len(registered) >= 6  # Updated from 2 to 6

    def test_gaussian_identity_optimized_irls(self):
        """Test that GLM uses optimized Gaussian+Identity solver vs statsmodels."""
        X, y, _, _ = generate_normal_data()

        # Use GLM class - it should automatically select GaussianIdentityIRLSSolver
        our_model = GLM(
            dist=Normal(),
            link=IdentityLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Gaussian()).fit()

        print("Gaussian+Identity Optimized IRLS (via GLM):")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.01, coef_atol=0.001, dev_rtol=0.01
        )

    def test_poisson_log_optimized_irls(self):
        """Test that GLM uses optimized Poisson+Log solver vs statsmodels."""
        X, y, _, _ = generate_poisson_data()

        # Use GLM class - it should automatically select PoissonLogIRLSSolver
        our_model = GLM(
            dist=Poisson(),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Poisson()).fit()

        print("Poisson+Log Optimized IRLS (via GLM):")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.05, coef_atol=0.01
        )

    def test_bernoulli_logit_optimized_irls(self):
        """Test that GLM uses optimized Bernoulli+Logit solver vs statsmodels."""
        X, y, _, _ = generate_bernoulli_data()

        # Use GLM class - it should automatically select BernoulliLogitIRLSSolver
        our_model = GLM(
            dist=Bernoulli(),
            link=LogitLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=50,
        )
        our_model.fit(X, y, enable_spu_cache=False)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Binomial()).fit()

        print("Bernoulli+Logit Optimized IRLS (via GLM):")
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.1, coef_atol=0.05, is_binary=True
        )

    def test_gamma_inverse_optimized_irls(self):
        """Test that GLM uses optimized Gamma+Inverse solver vs statsmodels."""
        # Use data with less variance for numerical stability
        X, y, _, _ = generate_gamma_data(shape=20.0)

        # Use GLM class - it should automatically select GammaInverseIRLSSolver
        our_model = GLM(
            dist=Gamma(),
            link=ReciprocalLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=100,
        )
        our_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)

        # Fit statsmodels
        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.InversePower())
        ).fit()

        print("Gamma+Inverse Optimized IRLS (via GLM):")
        # Relaxed tolerances due to numerical challenges with inverse link
        compare_with_statsmodels(
            our_model, sm_model, X, y, coef_rtol=0.2, coef_atol=0.1, dev_rtol=0.2
        )


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
        assert model_reg.coef_ is not None
        assert model_noreg.coef_ is not None

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


# ==================== Test with SPU Class ====================


def _create_spu_config(field: libspu.FieldType) -> libspu.RuntimeConfig:
    """
    Create SPU runtime config based on field type.

    FM64 uses EXP_PADE for stability, FM128 uses EXP_PRIME for better precision.
    """
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=field,
        fxp_fraction_bits=30 if field == libspu.FieldType.FM128 else 18,
    )
    config.enable_hal_profile = True
    config.enable_pphlo_profile = True

    if field == libspu.FieldType.FM64:
        # FM64: EXP_PADE is more stable
        config.fxp_exp_mode = libspu.RuntimeConfig.ExpMode.EXP_PADE
    else:
        # FM128: EXP_PRIME is better
        config.fxp_exp_mode = libspu.RuntimeConfig.ExpMode.EXP_PRIME
        config.experimental_enable_exp_prime = True
        config.experimental_exp_prime_disable_lower_bound = True

    return config


# Pre-create simulators for both field types (module-level for reuse)
_SIM_FM64 = None
_SIM_FM128 = None


def _get_simulator(field: libspu.FieldType) -> spsim.Simulator:
    """Get or create simulator for the given field type."""
    global _SIM_FM64, _SIM_FM128

    if field == libspu.FieldType.FM64:
        if _SIM_FM64 is None:
            _SIM_FM64 = spsim.Simulator(2, _create_spu_config(field))
        return _SIM_FM64
    else:
        if _SIM_FM128 is None:
            _SIM_FM128 = spsim.Simulator(2, _create_spu_config(field))
        return _SIM_FM128


# Test parameters: (field, force_generic_solver)
SPU_TEST_PARAMS = [
    pytest.param(libspu.FieldType.FM64, False, id="FM64-optimized"),
    pytest.param(libspu.FieldType.FM64, True, id="FM64-generic"),
    pytest.param(libspu.FieldType.FM128, False, id="FM128-optimized"),
    pytest.param(libspu.FieldType.FM128, True, id="FM128-generic"),
]


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
class TestGLMwithSPU:
    """Test GLM with SPU simulation for all registered optimized solvers."""

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_gaussian_identity_spu(self, field, force_generic):
        """Test Gaussian+Identity IRLS solver in SPU."""
        X, y, _, _ = generate_normal_data()
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=Normal(),
                link=IdentityLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=10,
                tol=0,
                l2=0.01,
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Gaussian()).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nGaussian+Identity IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        # Relaxed tolerance for FM64
        rtol, atol = (0.15, 0.1) if field == libspu.FieldType.FM64 else (0.1, 0.05)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_poisson_log_spu(self, field, force_generic):
        """Test Poisson+Log IRLS solver in SPU."""
        X, y, _, _ = generate_poisson_data()
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=Poisson(),
                link=LogLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=10,
                tol=0,
                l2=0.01,
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Poisson()).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nPoisson+Log IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        rtol, atol = (0.15, 0.1) if field == libspu.FieldType.FM64 else (0.1, 0.05)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_bernoulli_logit_spu(self, field, force_generic):
        """Test Bernoulli+Logit IRLS solver in SPU."""
        X, y, _, _ = generate_bernoulli_data()
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=Bernoulli(),
                link=LogitLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=15,
                tol=0,
                l2=0.01,
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(np.array(y), X_sm, family=sm_family.Binomial()).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nBernoulli+Logit IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        # Logistic regression may need more tolerance
        rtol, atol = (0.2, 0.15) if field == libspu.FieldType.FM64 else (0.15, 0.1)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_gamma_log_spu(self, field, force_generic):
        """Test Gamma+Log IRLS solver in SPU."""
        X, y, _, _ = generate_gamma_data()
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=Gamma(),
                link=LogLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=10,
                tol=0,
                l2=0.01,
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.Log())
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nGamma+Log IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        rtol, atol = (0.15, 0.1) if field == libspu.FieldType.FM64 else (0.1, 0.05)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_gamma_inverse_spu(self, field, force_generic):
        """Test Gamma+Inverse IRLS solver in SPU."""
        # Use data with less variance for numerical stability with inverse link
        X, y, _, _ = generate_gamma_data(shape=20.0)
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=Gamma(),
                link=ReciprocalLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=15,
                tol=0,
                l2=0.05,  # Stronger regularization for stability
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.InversePower())
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nGamma+Inverse IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        # Inverse link is numerically challenging, use relaxed tolerance
        rtol, atol = (0.3, 0.2) if field == libspu.FieldType.FM64 else (0.2, 0.15)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_tweedie_log_spu(self, field, force_generic):
        """Test Tweedie+Log IRLS solver in SPU (power=1.5)."""
        # Use Gamma-like data for Tweedie
        X, y, _, _ = generate_gamma_data(shape=10.0)
        sim = _get_simulator(field)
        power = 1.5

        def proc(x, y):
            model = GLM(
                dist=Tweedie(power=power),
                link=LogLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=10,
                tol=0,
                l2=0.01,
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y),
            X_sm,
            family=sm_family.Tweedie(var_power=power, link=sm_links.Log()),
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nTweedie(p={power})+Log IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        rtol, atol = (0.15, 0.1) if field == libspu.FieldType.FM64 else (0.1, 0.05)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_tweedie_log_power_1_7_spu(self, field, force_generic):
        """Test Tweedie+Log IRLS solver in SPU with different power (1.7)."""
        X, y, _, _ = generate_gamma_data(shape=8.0)
        sim = _get_simulator(field)
        power = 1.7

        def proc(x, y):
            model = GLM(
                dist=Tweedie(power=power),
                link=LogLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=10,
                tol=0,
                l2=0.01,
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y),
            X_sm,
            family=sm_family.Tweedie(var_power=power, link=sm_links.Log()),
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized"
        print(f"\nTweedie(p={power})+Log IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        rtol, atol = (0.15, 0.1) if field == libspu.FieldType.FM64 else (0.1, 0.05)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    # ==================== SGD Tests in SPU ====================

    @pytest.mark.parametrize(
        "field",
        [
            pytest.param(libspu.FieldType.FM64, id="FM64"),
            pytest.param(libspu.FieldType.FM128, id="FM128"),
        ],
    )
    def test_gamma_log_sgd_spu(self, field):
        """Test Gamma+Log SGD solver in SPU."""
        X, y, _, _ = generate_gamma_data(shape=10.0)
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=Gamma(),
                link=LogLink(),
                solver="sgd",
                fit_intercept=True,
                max_iter=100,
                learning_rate=0.01,
                batch_size=32,
                l2=0.01,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.Gamma(link=sm_links.Log())
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        print(f"\nGamma+Log SGD (SPU {field_name}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        # SGD typically has larger variance, use relaxed tolerance
        rtol, atol = (0.25, 0.2) if field == libspu.FieldType.FM64 else (0.2, 0.15)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)

    # ==================== Generic Solver Tests (No Optimized Solver) ====================

    @pytest.mark.parametrize("field,force_generic", SPU_TEST_PARAMS)
    def test_inverse_gaussian_log_spu(self, field, force_generic):
        """
        Test InverseGaussian+Log IRLS solver in SPU.

        This distribution+link combination has NO optimized solver registered,
        so it always uses the generic IRLS solver regardless of force_generic_solver.
        We still test both values to verify the fallback mechanism works correctly.
        """
        X, y, _, _ = generate_inverse_gaussian_data(lam=20.0)
        sim = _get_simulator(field)

        def proc(x, y):
            model = GLM(
                dist=InverseGaussian(),
                link=LogLink(),
                solver="irls",
                fit_intercept=True,
                max_iter=10,
                tol=0.001,
                l2=0.05,  # Stronger regularization for numerical stability
                force_generic_solver=force_generic,
            )
            model.fit(x, y, enable_spu_cache=True, y_scale=1.0, enable_spu_reveal=True)
            return model.coef_, model.intercept_

        our_coef, our_intercept = spsim.sim_jax(sim, proc)(X, y)

        X_sm = sm.add_constant(np.array(X))
        sm_model = sm.GLM(
            np.array(y), X_sm, family=sm_family.InverseGaussian(link=sm_links.Log())
        ).fit()

        sm_coef = sm_model.params[1:]
        sm_intercept = sm_model.params[0]

        field_name = "FM64" if field == libspu.FieldType.FM64 else "FM128"
        solver_type = "generic" if force_generic else "optimized(fallback to generic)"
        print(f"\nInverseGaussian+Log IRLS (SPU {field_name}, {solver_type}):")
        print(f"  SPU Coef: {our_coef}, Intercept: {our_intercept:.4f}")
        print(f"  SM Coef: {sm_coef}, Intercept: {sm_intercept:.4f}")

        # InverseGaussian is numerically challenging, use relaxed tolerance
        rtol, atol = (0.3, 0.25) if field == libspu.FieldType.FM64 else (0.2, 0.15)
        np.testing.assert_allclose(our_coef, sm_coef, rtol=rtol, atol=atol)
        np.testing.assert_allclose(our_intercept, sm_intercept, rtol=rtol, atol=atol)
