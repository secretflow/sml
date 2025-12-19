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

import jax
import jax.numpy as jnp
import numpy as np

from sml.linear_model.glm.core.distribution import Poisson
from sml.linear_model.glm.core.link import LogLink
from sml.linear_model.glm.formula.dispatch import register_formula
from sml.linear_model.glm.formula.optimized import PoissonLogFormula
from sml.linear_model.glm.model import GLM


class TestJAXGLM:
    def setup_method(self):
        # Generate synthetic Poisson data
        key = jax.random.PRNGKey(42)
        self.n_samples = 100
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

    def test_poisson_generic_jit(self):
        """Test Poisson GLM with Generic Formula (default) under JIT."""
        
        # Define a pure function to JIT
        @jax.jit
        def train_and_predict(X, y):
            # Use default link (Log) and generic formula
            model = GLM(dist=Poisson(), fit_intercept=True, tol=1e-6)
            model.fit(X, y)
            
            # Predict
            pred_mu = model.predict(X)
            return model.coef_, model.intercept_, pred_mu

        # Run JIT-compiled function
        coef, intercept, pred = train_and_predict(self.X, self.y)
        
        # Check correctness
        # Coefficients should be reasonably close to truth (given random noise)
        # Note: 100 samples is small, so tolerance is loose
        np.testing.assert_allclose(coef, self.true_coef, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(intercept, self.true_intercept, rtol=0.2, atol=0.2)
        
        print("\n[Generic] Coef:", coef)
        print("[Generic] Intercept:", intercept)

    def test_poisson_optimized_jit(self):
        """Test Poisson GLM with Optimized Formula under JIT."""
        
        # 1. Register the optimized formula explicitly
        register_formula(Poisson, LogLink, PoissonLogFormula())
        
        @jax.jit
        def train_and_predict(X, y):
            # Explicitly specify Poisson + LogLink to trigger optimized path
            model = GLM(dist=Poisson(), link=LogLink(), fit_intercept=True, tol=1e-6)
            model.fit(X, y)
            return model.coef_, model.intercept_

        coef, intercept = train_and_predict(self.X, self.y)
        
        # Should match the generic result closely (or be better/faster)
        np.testing.assert_allclose(coef, self.true_coef, rtol=0.2, atol=0.2)
        
        print("\n[Optimized] Coef:", coef)
        print("[Optimized] Intercept:", intercept)

if __name__ == "__main__":
    # Manually run if executed as script
    t = TestJAXGLM()
    t.setup_method()
    print("Running Generic JIT Test...")
    t.test_poisson_generic_jit()
    print("Running Optimized JIT Test...")
    t.test_poisson_optimized_jit()
