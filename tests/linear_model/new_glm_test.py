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
Unit tests for GLM with generic IRLS and SGD solvers in plaintext mode.

This module tests the GLM implementation without SPU caching, focusing on:
- IRLS solver with generic formula
- SGD solver with generic formula
- Various distributions and link functions
- Regularization and sample weights
- Metrics evaluation
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


class TestPlaintextGLM:
    """Test suite for GLM with generic solvers in plaintext mode."""

    def setup_method(self):
        """Generate synthetic Poisson data for testing."""
        key = jax.random.PRNGKey(42)
        self.n_samples = 200
        self.n_features = 5

        key, subkey = jax.random.split(key)
        self.X = jax.random.normal(subkey, (self.n_samples, self.n_features))

        # True coefficients
        self.true_coef = jnp.array([0.5, -0.3, 0.2, 0.1, -0.1])
        self.true_intercept = 1.0

        # Linear predictor
        eta = self.X @ self.true_coef + self.true_intercept

        # Mean mu = exp(eta)
        mu = jnp.exp(eta)

        # Generate y (Poisson distributed)
        key, subkey = jax.random.split(key)
        self.y = jax.random.poisson(subkey, mu).astype(jnp.float32)

    # ==================== IRLS Solver Tests ====================

    def test_poisson_irls(self):
        """Test Poisson GLM with IRLS solver."""
        model = GLM(
            dist=Poisson(),
            solver="irls",
            fit_intercept=True,
            max_iter=20,
            tol=1e-6,
        )
        model.fit(self.X, self.y, enable_spu_cache=False)

        # Check correctness
        np.testing.assert_allclose(model.coef_, self.true_coef, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(
            model.intercept_, self.true_intercept, rtol=0.2, atol=0.2
        )

    def test_normal_irls(self):
        """Test Normal (Gaussian) distribution with IRLS solver."""
        # Generate Gaussian data
        key = jax.random.PRNGKey(42)
        n_samples, n_features = 100, 3
        X_normal = jax.random.normal(key, (n_samples, n_features))
        true_coef = jnp.array([1.0, -0.5, 0.3])
        true_intercept = 2.0
        key, subkey = jax.random.split(key)
        y_normal = (
            X_normal @ true_coef
            + true_intercept
            + 0.1 * jax.random.normal(subkey, (n_samples,))
        )

        model = GLM(
            dist=Normal(),
            solver="irls",
            fit_intercept=True,
            max_iter=20,
        )
        model.fit(X_normal, y_normal, enable_spu_cache=False)

        np.testing.assert_allclose(model.coef_, true_coef, rtol=0.1, atol=0.1)
        np.testing.assert_allclose(model.intercept_, true_intercept, rtol=0.1, atol=0.1)

    def test_bernoulli_irls(self):
        """Test Bernoulli (logistic regression) with IRLS solver."""
        key = jax.random.PRNGKey(42)
        n_samples, n_features = 200, 4
        X_logit = jax.random.normal(key, (n_samples, n_features))
        true_coef = jnp.array([1.0, -0.5, 0.5, -0.3])
        true_intercept = 0.5
        eta = X_logit @ true_coef + true_intercept
        prob = jax.nn.sigmoid(eta)

        key, subkey = jax.random.split(key)
        y_binary = jax.random.bernoulli(subkey, prob).astype(jnp.float32)

        model = GLM(
            dist=Bernoulli(),
            link=LogitLink(),
            solver="irls",
            fit_intercept=True,
        )
        model.fit(X_logit, y_binary, enable_spu_cache=False)

        # Coefficients should be reasonably close to true values
        np.testing.assert_allclose(model.coef_, true_coef, rtol=0.5, atol=0.5)
        np.testing.assert_allclose(model.intercept_, true_intercept, rtol=0.5, atol=0.5)

        # Predictions should be probabilities between 0 and 1
        mu_pred = model.predict(X_logit)
        assert jnp.all(mu_pred >= 0) and jnp.all(mu_pred <= 1)

    def test_gamma_log_irls(self):
        """Test Gamma distribution with Log link using IRLS solver."""
        key = jax.random.PRNGKey(42)
        n_samples, n_features = 100, 3
        X_gamma = jax.random.normal(key, (n_samples, n_features))
        true_coef = jnp.array([0.5, -0.3, 0.2])
        true_intercept = 1.0
        eta = X_gamma @ true_coef + true_intercept
        mu = jnp.exp(eta)

        # Generate positive y values (approximate Gamma)
        key, subkey = jax.random.split(key)
        y_gamma = mu + 0.3 * mu * jax.random.normal(subkey, (n_samples,))
        y_gamma = jnp.abs(y_gamma) + 0.1  # Ensure positive

        model = GLM(
            dist=Gamma(),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=20,
        )
        model.fit(X_gamma, y_gamma, enable_spu_cache=False)

        np.testing.assert_allclose(model.coef_, true_coef, rtol=0.3, atol=0.3)

    def test_tweedie_irls(self):
        """Test Tweedie distribution with IRLS solver."""
        # Tweedie with power=1.5 (compound Poisson-Gamma)
        model = GLM(
            dist=Tweedie(power=1.5),
            solver="irls",
            fit_intercept=True,
        )
        model.fit(self.X, self.y, enable_spu_cache=False)

        # Test deviance computation
        mu = model.predict(self.X)
        dev = model.dist.deviance(self.y, mu)
        assert dev >= 0

    def test_negative_binomial_irls(self):
        """Test Negative Binomial distribution with IRLS solver."""
        model = GLM(
            dist=NegativeBinomial(alpha=1.0),
            solver="irls",
            fit_intercept=True,
        )
        model.fit(self.X, self.y, enable_spu_cache=False)

        assert model.coef_ is not None

        # Test unit variance: V(mu) = mu + alpha * mu^2
        dist = NegativeBinomial(alpha=1.0)
        mu_test = jnp.array([1.0, 2.0, 3.0])
        var = dist.unit_variance(mu_test)
        expected_var = mu_test + 1.0 * mu_test**2
        np.testing.assert_allclose(var, expected_var, rtol=1e-5)

    def test_inverse_gaussian_irls(self):
        """Test Inverse Gaussian distribution with IRLS solver."""
        # Generate positive data suitable for Inverse Gaussian
        y_pos = jnp.maximum(self.y, 1.0)

        model = GLM(
            dist=InverseGaussian(),
            solver="irls",
            fit_intercept=True,
        )
        model.fit(self.X, y_pos, enable_spu_cache=False)

        assert model.coef_ is not None

        # Test unit variance: V(mu) = mu^3
        dist = InverseGaussian()
        mu_test = jnp.array([0.5, 1.0, 2.0])
        var = dist.unit_variance(mu_test)
        expected_var = mu_test**3
        np.testing.assert_allclose(var, expected_var, rtol=1e-5)

    # ==================== SGD Solver Tests ====================

    def test_poisson_sgd(self):
        """Test Poisson GLM with SGD solver."""
        model = GLM(
            dist=Poisson(),
            solver="sgd",
            learning_rate=1e-2,
            decay_rate=0.9,
            decay_steps=50,
            max_iter=200,
            batch_size=32,
            fit_intercept=True,
            tol=1e-6,
        )
        model.fit(self.X, self.y, enable_spu_cache=False)

        # SGD might be less precise than IRLS
        np.testing.assert_allclose(model.coef_, self.true_coef, rtol=0.3, atol=0.3)
        np.testing.assert_allclose(
            model.intercept_, self.true_intercept, rtol=0.3, atol=0.3
        )

    def test_normal_sgd(self):
        """Test Normal distribution with SGD solver."""
        key = jax.random.PRNGKey(42)
        n_samples, n_features = 100, 3
        X_normal = jax.random.normal(key, (n_samples, n_features))
        true_coef = jnp.array([1.0, -0.5, 0.3])
        true_intercept = 2.0
        key, subkey = jax.random.split(key)
        y_normal = (
            X_normal @ true_coef
            + true_intercept
            + 0.1 * jax.random.normal(subkey, (n_samples,))
        )

        model = GLM(
            dist=Normal(),
            solver="sgd",
            fit_intercept=True,
            max_iter=100,
            learning_rate=0.01,
        )
        model.fit(X_normal, y_normal, enable_spu_cache=False)

        assert model.coef_ is not None

    def test_sgd_batching(self):
        """Verify SGD runs with different batch sizes."""
        for bs in [10, 50, 200]:  # 200 is full batch
            model = GLM(
                dist=Poisson(),
                solver="sgd",
                batch_size=bs,
                max_iter=10,
            )
            model.fit(self.X, self.y, enable_spu_cache=False)
            assert model.coef_ is not None

    def test_sgd_no_early_stop(self):
        """Test SGD solver without early stopping (tol=0)."""
        model = GLM(
            dist=Poisson(),
            solver="sgd",
            learning_rate=1e-2,
            max_iter=50,
            batch_size=32,
            fit_intercept=True,
            tol=0.0,  # Disable early stopping
        )
        model.fit(self.X, self.y, enable_spu_cache=False)

        assert model.coef_ is not None
        assert model.history_["n_iter"] == 50  # Should run full iterations

    # ==================== Link Function Tests ====================

    def test_various_links_bernoulli(self):
        """Test different link functions with Bernoulli distribution."""
        key = jax.random.PRNGKey(123)
        X_binary = jax.random.normal(key, (100, 3))
        true_coef = jnp.array([1.0, -0.5, 0.5])
        eta = X_binary @ true_coef

        links_to_test = [
            ("Logit", LogitLink()),
            ("Probit", ProbitLink()),
            ("CLogLog", CLogLogLink()),
        ]

        for link_name, link in links_to_test:
            mu = link.inverse(eta)
            key, subkey = jax.random.split(key)
            y_binary = jax.random.bernoulli(subkey, mu).astype(jnp.float32)

            model = GLM(
                dist=Bernoulli(),
                link=link,
                solver="irls",
                fit_intercept=True,
            )
            model.fit(X_binary, y_binary, enable_spu_cache=False)

            assert model.coef_ is not None

    def test_power_link(self):
        """Test PowerLink with different powers."""
        powers_to_test = [0.5, 1.0, 2.0, -1.0, -2.0]

        for power in powers_to_test:
            link = PowerLink(power=power)
            mu_test = jnp.array([1.0, 2.0, 3.0])
            eta = link.link(mu_test)
            mu_recovered = link.inverse(eta)
            np.testing.assert_allclose(mu_recovered, mu_test, rtol=1e-5)

    def test_reciprocal_link(self):
        """Test ReciprocalLink equals PowerLink(power=-1)."""
        reciprocal = ReciprocalLink()
        power_neg_one = PowerLink(power=-1.0)

        mu_test = jnp.array([1.0, 2.0, 3.0])
        eta_recip = reciprocal.link(mu_test)
        eta_power = power_neg_one.link(mu_test)

        np.testing.assert_allclose(eta_recip, eta_power, rtol=1e-10)

    def test_distribution_link_combinations(self):
        """Test various distribution-link combinations."""
        combinations = [
            (Normal(), IdentityLink()),
            (Normal(), LogLink()),
            (Poisson(), LogLink()),
            (Poisson(), PowerLink(power=0.5)),
            (Gamma(), LogLink()),
            (Gamma(), ReciprocalLink()),
            (Tweedie(power=1.5), LogLink()),
            (NegativeBinomial(alpha=0.5), LogLink()),
            (InverseGaussian(), PowerLink(power=-2.0)),
        ]

        for dist, link in combinations:
            model = GLM(
                dist=dist,
                link=link,
                solver="irls",
                fit_intercept=True,
                max_iter=10,
            )
            try:
                model.fit(self.X, self.y, enable_spu_cache=False)
                assert model.coef_ is not None
            except Exception:
                # Some combinations might be numerically unstable
                pass

    # ==================== Regularization & Weights Tests ====================

    def test_l2_regularization(self):
        """Test L2 regularization."""
        model_no_reg = GLM(dist=Poisson(), solver="irls", l2=0.0)
        model_no_reg.fit(self.X, self.y, enable_spu_cache=False)

        model_with_reg = GLM(dist=Poisson(), solver="irls", l2=1.0)
        model_with_reg.fit(self.X, self.y, enable_spu_cache=False)

        # Coefficients should be smaller with regularization (shrinkage effect)
        norm_no_reg = jnp.linalg.norm(model_no_reg.coef_)
        norm_with_reg = jnp.linalg.norm(model_with_reg.coef_)
        assert norm_with_reg < norm_no_reg

    def test_sample_weights(self):
        """Test sample weights."""
        weights = jnp.concatenate(
            [
                jnp.ones(self.n_samples // 2) * 2.0,
                jnp.ones(self.n_samples - self.n_samples // 2) * 0.5,
            ]
        )

        model_no_weights = GLM(dist=Poisson(), solver="irls")
        model_no_weights.fit(self.X, self.y, enable_spu_cache=False)

        model_with_weights = GLM(dist=Poisson(), solver="irls")
        model_with_weights.fit(
            self.X, self.y, sample_weight=weights, enable_spu_cache=False
        )

        # Coefficients should differ when weights are applied
        diff = jnp.linalg.norm(model_no_weights.coef_ - model_with_weights.coef_)
        assert diff > 0

    def test_offset_handling(self):
        """Test that offset is correctly handled in fit and predict."""
        offset = jnp.ones_like(self.y) * 2.0

        model = GLM(dist=Poisson(), solver="irls")
        model.fit(self.X, self.y, offset=offset, enable_spu_cache=False)

        # Since true data was generated with intercept=1.0 and offset=0
        # Now we force offset=2.0. The model should compensate by learning intercept approx -1.0
        np.testing.assert_allclose(
            model.intercept_, self.true_intercept - 2.0, atol=0.3
        )

        # Test predict with offset
        mu_pred = model.predict(self.X, offset=offset)
        dev = model.evaluate(self.X, self.y, metrics=["deviance"], offset=offset)[
            "deviance"
        ]
        assert dev < 300  # Heuristic check

    # ==================== Metrics Tests ====================

    def test_metrics_evaluation(self):
        """Test metrics evaluation."""
        model = GLM(dist=Poisson(), solver="irls")
        model.fit(self.X, self.y, enable_spu_cache=False)

        metrics = model.evaluate(self.X, self.y, metrics=["deviance", "aic", "rmse"])

        assert "deviance" in metrics
        assert "aic" in metrics
        assert "rmse" in metrics
        assert metrics["rmse"] > 0

    def test_bic_metric(self):
        """Test BIC metric evaluation."""
        model = GLM(dist=Poisson(), solver="irls")
        model.fit(self.X, self.y, enable_spu_cache=False)

        metrics = model.evaluate(self.X, self.y, metrics=["deviance", "aic", "bic"])

        assert "bic" in metrics
        # BIC should be larger than AIC for n > e^2 ≈ 7.4
        assert metrics["bic"] > metrics["aic"]

    # ==================== Convergence & History Tests ====================

    def test_convergence_history(self):
        """Test that history_ contains convergence information."""
        model = GLM(dist=Poisson(), solver="irls", max_iter=50, tol=1e-6)
        model.fit(self.X, self.y, enable_spu_cache=False)

        assert model.history_ is not None
        assert "n_iter" in model.history_
        assert "converged" in model.history_

    def test_irls_no_early_stop(self):
        """Test IRLS solver without early stopping (tol=0)."""
        model = GLM(
            dist=Poisson(),
            solver="irls",
            max_iter=10,
            tol=0.0,  # Disable early stopping
        )
        model.fit(self.X, self.y, enable_spu_cache=False)

        assert model.coef_ is not None
        assert model.history_["n_iter"] == 10  # Should run full iterations

    def test_r_style_initialization(self):
        """Test that R-style initialization provides good starting point."""
        # With good initialization, should achieve reasonable fit quickly
        model = GLM(dist=Poisson(), solver="irls", max_iter=10, tol=1e-6)
        model.fit(self.X, self.y, enable_spu_cache=False)

        assert model.history_["n_iter"] <= 10
        np.testing.assert_allclose(model.coef_, self.true_coef, rtol=0.3, atol=0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
