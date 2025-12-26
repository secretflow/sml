# Copyright 2025 Ant Group Co., Ltd.
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
GLM (Generalized Linear Model) emulation test.

This module tests GLM with Gamma distribution and Log link function
in SPU environment.
"""

import jax
import jax.numpy as jnp
import numpy as np

import emulations.utils.emulation as emulation
from sml.linear_model.glm import GLM
from sml.linear_model.glm.core.distribution import Gamma
from sml.linear_model.glm.core.link import LogLink


def generate_gamma_data(seed=42, n_samples=3000, n_features=30, shape=5.0):
    """
    Generate Gamma distributed data for GLM testing.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features.
    shape : float
        Shape parameter for Gamma distribution.

    Returns
    -------
    X : jnp.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target variable following Gamma distribution.
    true_coef : jnp.ndarray
        True coefficients used for data generation.
    true_intercept : float
        True intercept used for data generation.
    """
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Generate features with smaller scale for numerical stability
    X = jax.random.normal(k1, (n_samples, n_features)) * 0.5

    # Dynamically generate coefficients (smaller range for stability)
    true_coef = jax.random.uniform(k3, (n_features,), minval=-0.5, maxval=0.5)
    true_intercept = 1.5

    # Linear predictor
    eta = X @ true_coef + true_intercept
    # Mean (using log link, so mu = exp(eta))
    mu = jnp.exp(eta)

    # Gamma: mean = shape * scale, so scale = mu / shape
    y = jax.random.gamma(k2, shape, (n_samples,)) * (mu / shape)

    return X, y, true_coef, true_intercept


def emul_GLM_Gamma_Log(emulator: emulation.Emulator):
    """
    Emulation test for GLM with Gamma distribution and Log link function.

    This test:
    1. Generates Gamma-distributed data
    2. Fits a GLM model in SPU environment
    3. Compares predictions and coefficients with plaintext results
    """

    def proc_fit(X, y):
        """Fit GLM model and return coefficients and predictions."""
        model = GLM(
            dist=Gamma(),
            link=LogLink(),
            solver="irls",
            fit_intercept=True,
            max_iter=30,
            tol=1e-6,
        )
        model.fit(X, y, enable_spu_cache=True, y_scale=1.0)

        # Get predictions
        mu_pred = model.predict(X)

        return model.coef_, model.intercept_, mu_pred

    # Generate test data
    print("Generating Gamma distributed data (3000 samples, 30 features)...")
    X, y, true_coef, true_intercept = generate_gamma_data(
        seed=42, n_samples=3000, n_features=30, shape=5.0
    )

    # Convert to numpy for sealing
    X_np = np.array(X)
    y_np = np.array(y)

    # Run plaintext version first for comparison
    print("\n[Plaintext] Fitting GLM (Gamma + Log)...")
    plain_model = GLM(
        dist=Gamma(),
        link=LogLink(),
        solver="irls",
        fit_intercept=True,
        max_iter=30,
        tol=1e-6,
    )
    plain_model.fit(X, y, enable_spu_cache=False, y_scale=1.0)
    plain_mu = plain_model.predict(X)
    plain_deviance = float(plain_model.dist.deviance(y, plain_mu))
    assert plain_model.coef_ is not None

    print(f"  Plaintext Deviance: {plain_deviance:.4f}")
    print(f"  Plaintext Coef (first 5): {plain_model.coef_[:5]}")
    print(f"  Plaintext Intercept: {plain_model.intercept_:.4f}")

    # Mark data to be protected in SPU
    X_spu, y_spu = emulator.seal(X_np, y_np)

    # Run in SPU
    print("\n[SPU] Fitting GLM (Gamma + Log)...")
    spu_coef, spu_intercept, spu_mu = emulator.run(proc_fit)(X_spu, y_spu)

    # Convert results
    spu_coef = jnp.array(spu_coef)
    spu_intercept = float(spu_intercept)
    spu_mu = jnp.array(spu_mu)
    spu_deviance = float(Gamma().deviance(y, spu_mu))

    print(f"  SPU Deviance: {spu_deviance:.4f}")
    print(f"  SPU Coef (first 5): {spu_coef[:5]}")
    print(f"  SPU Intercept: {spu_intercept:.4f}")

    # Compare results
    print("\n[Comparison]")
    assert plain_model.intercept_ is not None
    coef_diff = jnp.abs(plain_model.coef_ - spu_coef)
    intercept_diff = abs(plain_model.intercept_ - spu_intercept)
    deviance_diff = abs(plain_deviance - spu_deviance)

    print(f"  Max Coef Difference: {jnp.max(coef_diff):.6f}")
    print(f"  Mean Coef Difference: {jnp.mean(coef_diff):.6f}")
    print(f"  Intercept Difference: {intercept_diff:.6f}")
    print(f"  Deviance Difference: {deviance_diff:.4f}")

    # Assertions for validation
    assert jnp.max(coef_diff) < 0.5, "Coefficient difference too large"
    assert intercept_diff < 0.5, "Intercept difference too large"
    # Deviance relative error should be small
    rel_dev_error = deviance_diff / plain_deviance
    print(f"  Deviance Relative Error: {rel_dev_error:.4f}")
    assert rel_dev_error < 0.1, "Deviance relative error too large"

    print("\n[PASSED] GLM Gamma + Log emulation test passed!")


def main(
    cluster_config: str = emulation.CLUSTER_ABY3_3PC,
    mode: emulation.Mode = emulation.Mode.MULTIPROCESS,
    bandwidth: int = 300,
    latency: int = 20,
):
    """Main entry point for GLM emulation tests."""
    with emulation.start_emulator(
        cluster_config,
        mode,
        bandwidth,
        latency,
    ) as emulator:
        emul_GLM_Gamma_Log(emulator)


if __name__ == "__main__":
    main()
