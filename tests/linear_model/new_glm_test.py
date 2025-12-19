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
from sml.linear_model.glm.core.distribution import Poisson
from sml.linear_model.glm.core.link import LogLink
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

    def test_poisson_sgd_jit(self):
        """Test Poisson GLM with SGD Solver."""
        
        print("\n--- Testing SGD Solver ---")
        
        # Need careful tuning for SGD on Poisson
        # Smaller LR and more epochs
        lr = 1e-3
        epochs = 200
        batch_size = 32
        
        start_time = time.time()
        
        @jax.jit
        def train_and_predict(X, y):
            model = GLM(
                dist=Poisson(), 
                solver='sgd', 
                learning_rate=lr, 
                max_iter=epochs, 
                batch_size=batch_size,
                fit_intercept=True, 
                tol=1e-6
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

if __name__ == "__main__":
    t = TestJAXGLM()
    t.setup_method()
    t.test_poisson_generic_jit_naive_inv()
    t.test_poisson_sgd_jit()
    t.test_metrics()
    t.test_sgd_batching()