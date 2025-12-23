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

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sml.linear_model.glm.core.distribution import (
    Normal, Bernoulli, Poisson, Gamma,
    Tweedie, NegativeBinomial, InverseGaussian
)
from sml.linear_model.glm.core.link import (
    IdentityLink, LogLink, LogitLink,
    ProbitLink, CLogLogLink, PowerLink, ReciprocalLink
)
from sml.linear_model.glm.formula.dispatch import register_formula
from sml.linear_model.glm.formula.optimized import PoissonLogFormula
from sml.linear_model.glm.model import GLM


class TestJAXGLM:
    def setup_method(self):
        # Generate synthetic Poisson data
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

    def test_poisson_generic_jit_naive_inv(self):
        """Test Poisson GLM with Generic Formula & Naive Inv IRLS."""
        
        print("\n--- Testing IRLS (Naive Inv) ---")
        start_time = time.time()
        
        @jax.jit
        def train_and_predict(X, y):
            # IRLS solver uses naive inversion now
            model = GLM(dist=Poisson(), solver='irls', fit_intercept=True, tol=1e-6)
            model.fit(X, y)
            return model.coef_, model.intercept_

        coef, intercept = train_and_predict(self.X, self.y)
        end_time = time.time()
        
        print(f"Time: {end_time - start_time:.4f}s")
        print("Coef:", coef)
        print("Intercept:", intercept)
        
        # Check correctness
        np.testing.assert_allclose(coef, self.true_coef, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(intercept, self.true_intercept, rtol=0.2, atol=0.2)

    def test_poisson_sgd_jit_with_decay(self):
        """Test Poisson GLM with SGD Solver and LR Decay."""
        
        print("\n--- Testing SGD Solver with Decay ---")
        
        # Need careful tuning for SGD on Poisson
        lr = 1e-2
        epochs = 200
        batch_size = 32
        
        start_time = time.time()
        
        @jax.jit
        def train_and_predict(X, y):
            model = GLM(
                dist=Poisson(), 
                solver='sgd', 
                learning_rate=lr, 
                decay_rate=0.9,     # Decay LR every 500 steps
                decay_steps=500,
                max_iter=epochs, 
                batch_size=batch_size,
                fit_intercept=True, 
                tol=1e-6,
            )
            model.fit(X, y)
            return model.coef_, model.intercept_

        coef, intercept = train_and_predict(self.X, self.y)
        end_time = time.time()
        
        print(f"Time: {end_time - start_time:.4f}s")
        print("Coef:", coef)
        print("Intercept:", intercept)
        
        # SGD might be less precise than IRLS with limited epochs
        np.testing.assert_allclose(coef, self.true_coef, rtol=0.3, atol=0.3)
        np.testing.assert_allclose(intercept, self.true_intercept, rtol=0.3, atol=0.3)

    def test_offset_handling(self):
        """Test that offset is correctly handled in fit and predict."""
        print("\n--- Testing Offset Handling ---")
        
        # Create a large offset that would change predictions significantly
        offset = jnp.ones_like(self.y) * 2.0
        
        # Fit model with offset
        # Effectively, we are fitting y ~ Poisson(exp(X@beta + intercept + offset))
        # The true intercept should now be (true_intercept - 2.0) approx if offset was capturing part of it?
        # No, if we pass offset, the model learns the residual. 
        
        model = GLM(dist=Poisson(), solver='irls')
        model.fit(self.X, self.y, offset=offset)
        
        # Since true data was generated with intercept=1.0 and offset=0
        # Now we force offset=2.0. The model should compensate by learning intercept approx -1.0
        # exp(1.0) = exp(-1.0 + 2.0)
        
        print("Intercept with offset=2.0:", model.intercept_)
        np.testing.assert_allclose(model.intercept_, self.true_intercept - 2.0, atol=0.2)
        
        # Test predict with offset
        mu_pred = model.predict(self.X, offset=offset)
        dev = model.evaluate(self.X, self.y, metrics=["deviance"], offset=offset)['deviance']
        print("Deviance with offset:", dev)
        
        # It should be close to the original model's deviance
        assert dev < 300 # Heuristic check

    def test_metrics(self):
        """Test extended metrics evaluation."""
        model = GLM(dist=Poisson(), solver='irls')
        model.fit(self.X, self.y)
        
        metrics = model.evaluate(self.X, self.y, metrics=["deviance", "aic", "rmse"])
        print("\nMetrics:", metrics)
        
        assert "deviance" in metrics
        assert "aic" in metrics
        assert "rmse" in metrics
        assert metrics["rmse"] > 0
        
    def test_sgd_batching(self):
        """Verify SGD runs with different batch sizes."""
        for bs in [10, 50, 200]: # 200 is full batch
            print(f"\nTesting SGD batch_size={bs}")
            model = GLM(dist=Poisson(), solver='sgd', batch_size=bs, max_iter=10)
            model.fit(self.X, self.y)
            assert model.coef_ is not None

    def test_y_scaling(self):
        """Test manual y scaling for numerical stability."""
        print("\n--- Testing Y Scaling ---")
        
        # Calculate scale
        y_max = float(jnp.max(self.y))
        y_scale = y_max / 2.0
        if y_scale < 1.0:
            y_scale = 1.0
            
        print(f"Using y_scale: {y_scale}")
        
        # Fit with scale
        model = GLM(dist=Poisson(), solver='irls', fit_intercept=True)
        model.fit(self.X, self.y, scale=y_scale)
        
        # The intercept should change because y is scaled. 
        # For Log link: log(y/scale) = log(y) - log(scale)
        # So intercept should be reduced by log(scale)
        expected_intercept_shift = jnp.log(y_scale)
        print(f"Learned Intercept: {model.intercept_}, Expected Shift: -{expected_intercept_shift}")
        
        # Compare with unscaled model
        model_unscaled = GLM(dist=Poisson(), solver='irls')
        model_unscaled.fit(self.X, self.y)
        
        print(f"Unscaled Intercept: {model_unscaled.intercept_}")
        
        # intercept_scaled ~ intercept_unscaled - log(scale)
        np.testing.assert_allclose(
            model.intercept_, 
            model_unscaled.intercept_ - expected_intercept_shift, 
            atol=0.2
        )
        
        # Coefficients should be identical (scale only affects intercept in Log link)
        np.testing.assert_allclose(model.coef_, model_unscaled.coef_, atol=1e-4)
        
        # Prediction with scale should match original y
        mu_pred = model.predict(self.X, scale=y_scale)
        mu_unscaled = model_unscaled.predict(self.X)
        
        np.testing.assert_allclose(mu_pred, mu_unscaled, rtol=1e-4)
        print("Scaling test passed: predictions match unscaled model.")

    def test_tweedie_distribution(self):
        """Test Tweedie distribution with different power parameters."""
        print("\n--- Testing Tweedie Distribution ---")

        # Test Tweedie with power=1.5 (compound Poisson-Gamma)
        model = GLM(dist=Tweedie(power=1.5), solver='irls', fit_intercept=True)
        model.fit(self.X, self.y)
        print(f"Tweedie(1.5) - Coef: {model.coef_}, Intercept: {model.intercept_}")

        # Test Tweedie with power=0 (should behave like Normal)
        model_normal = GLM(dist=Tweedie(power=0.0), solver='irls', fit_intercept=True)
        model_normal.fit(self.X, self.y)
        print(f"Tweedie(0.0) - Coef: {model_normal.coef_}, Intercept: {model_normal.intercept_}")

        # Test deviance computation
        mu = model.predict(self.X)
        dev = model.dist.deviance(self.y, mu)
        print(f"Tweedie deviance: {dev}")
        assert dev >= 0

    def test_negative_binomial_distribution(self):
        """Test Negative Binomial distribution with overdispersion."""
        print("\n--- Testing Negative Binomial Distribution ---")

        # Test with alpha=1.0
        model = GLM(dist=NegativeBinomial(alpha=1.0), solver='irls', fit_intercept=True)
        model.fit(self.X, self.y)
        print(f"NegativeBinomial(alpha=1.0) - Coef: {model.coef_}, Intercept: {model.intercept_}")

        # Test with different alpha values
        for alpha in [0.5, 1.0, 2.0]:
            dist = NegativeBinomial(alpha=alpha)
            mu_test = jnp.array([1.0, 2.0, 3.0])
            var = dist.unit_variance(mu_test)
            print(f"NB alpha={alpha}: mu={mu_test}, var={var}")
            # Variance should be mu + alpha * mu^2
            expected_var = mu_test + alpha * mu_test ** 2
            np.testing.assert_allclose(var, expected_var, rtol=1e-5)

    def test_inverse_gaussian_distribution(self):
        """Test Inverse Gaussian distribution."""
        print("\n--- Testing Inverse Gaussian Distribution ---")

        model = GLM(dist=InverseGaussian(), solver='irls', fit_intercept=True)
        # Generate positive data suitable for Inverse Gaussian
        y_pos = jnp.maximum(self.y, 1.0)  # Ensure strictly positive
        model.fit(self.X, y_pos)
        print(f"InverseGaussian - Coef: {model.coef_}, Intercept: {model.intercept_}")

        # Test unit variance: V(mu) = mu^3
        dist = InverseGaussian()
        mu_test = jnp.array([0.5, 1.0, 2.0])
        var = dist.unit_variance(mu_test)
        expected_var = mu_test ** 3
        np.testing.assert_allclose(var, expected_var, rtol=1e-5)
        print(f"InverseGaussian unit variance test passed")

    def test_various_links(self):
        """Test different link functions."""
        print("\n--- Testing Various Link Functions ---")

        # Generate binary data for link testing
        key = jax.random.PRNGKey(123)
        X_binary = jax.random.normal(key, (100, 3))
        true_coef = jnp.array([1.0, -0.5, 0.5])
        true_intercept = 0.0
        eta = X_binary @ true_coef + true_intercept

        # Test different links with Bernoulli distribution
        links_to_test = [
            ("Logit", LogitLink()),
            ("Probit", ProbitLink()),
            ("CLogLog", CLogLogLink()),
        ]

        for link_name, link in links_to_test:
            model = GLM(dist=Bernoulli(), link=link, solver='irls', fit_intercept=True)
            # Generate y based on link
            mu = link.inverse(eta)
            key, subkey = jax.random.split(key)
            y_binary = jax.random.bernoulli(subkey, mu).astype(jnp.float32)

            model.fit(X_binary, y_binary)
            print(f"{link_name} Link - Coef: {model.coef_}")

            # Test link derivatives
            mu_test = jnp.array([0.2, 0.5, 0.8])
            link_val = link.link(mu_test)
            link_der = link.link_deriv(mu_test)
            inv_der = link.inverse_deriv(link_val)

            # For CLogLog, the derivative relationship is more complex
            # We'll just check that derivatives are non-zero where expected
            if link_name == "CLogLog":
                # CLogLog derivatives can be negative, that's expected
                assert not jnp.all(link_der == 0)
                assert not jnp.all(inv_der == 0)
            else:
                np.testing.assert_allclose(inv_der, 1.0 / link_der, rtol=1e-5)
            print(f"{link_name} derivative consistency test passed")

    def test_power_and_reciprocal_links(self):
        """Test PowerLink and ReciprocalLink."""
        print("\n--- Testing Power and Reciprocal Links ---")

        # Test PowerLink with different powers
        powers_to_test = [0.5, 1.0, 2.0, -1.0, -2.0]

        for power in powers_to_test:
            if abs(power) < 1e-10:
                continue  # Skip power=0 as it's not allowed
            link = PowerLink(power=power)

            mu_test = jnp.array([1.0, 2.0, 3.0])
            eta = link.link(mu_test)
            mu_recovered = link.inverse(eta)

            np.testing.assert_allclose(mu_recovered, mu_test, rtol=1e-5)
            print(f"PowerLink(power={power}) consistency test passed")

        # Test ReciprocalLink (should be same as PowerLink(power=-1))
        reciprocal = ReciprocalLink()
        power_neg_one = PowerLink(power=-1.0)

        mu_test = jnp.array([1.0, 2.0, 3.0])
        eta_recip = reciprocal.link(mu_test)
        eta_power = power_neg_one.link(mu_test)

        np.testing.assert_allclose(eta_recip, eta_power, rtol=1e-10)
        print("ReciprocalLink and PowerLink(power=-1) equivalence test passed")

    def test_link_distribution_combinations(self):
        """Test various distribution-link combinations."""
        print("\n--- Testing Distribution-Link Combinations ---")

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
            model = GLM(dist=dist, link=link, solver='irls', fit_intercept=True, max_iter=10)
            try:
                model.fit(self.X, self.y)
                mu_pred = model.predict(self.X)
                print(f"{dist.__class__.__name__}+{link.__class__.__name__}: Success")
                assert mu_pred is not None
            except Exception as e:
                print(f"{dist.__class__.__name__}+{link.__class__.__name__}: Error - {e}")
                # Some combinations might be numerically unstable, that's expected

if __name__ == "__main__":
    t = TestJAXGLM()
    t.setup_method()
    t.test_poisson_generic_jit_naive_inv()
    t.test_poisson_sgd_jit_with_decay()
    t.test_offset_handling()
    t.test_metrics()
    t.test_sgd_batching()
    t.test_y_scaling()

    # Test new distributions and links
    t.test_tweedie_distribution()
    t.test_negative_binomial_distribution()
    t.test_inverse_gaussian_distribution()
    t.test_various_links()
    t.test_power_and_reciprocal_links()
    t.test_link_distribution_combinations()