# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np

import emulations.utils.emulation as emulation
from sml.stats import vif
from sml.utils.extmath import standardize


def _statsmodels_vif(data: jnp.ndarray, standardized: bool):
    from statsmodels.stats.outliers_influence import variance_inflation_factor as sm_vif

    if not standardized:
        data = standardize(data)

    cols = data.shape[1]
    ret = np.array([sm_vif(data, i) for i in range(cols)])
    return ret


def emul_vif(emulator: emulation.Emulator):
    # Test with synthetic data
    seed = 42
    key = jax.random.PRNGKey(seed)

    # Generate correlated data
    n_samples, n_features = 1000, 5

    # Create a correlation matrix
    mean = jnp.zeros(n_features)
    cov = jnp.array(
        [
            [1.0, 0.8, 0.3, 0.1, 0.2],
            [0.8, 1.0, 0.4, 0.2, 0.1],
            [0.3, 0.4, 1.0, 0.5, 0.3],
            [0.1, 0.2, 0.5, 1.0, 0.6],
            [0.2, 0.1, 0.3, 0.6, 1.0],
        ]
    )

    # Generate multivariate normal data
    X = jax.random.multivariate_normal(key, mean, cov, shape=(n_samples,))

    ref_result = _statsmodels_vif(np.array(X), standardized=False)

    # Run SPU computation with non-standardized data
    spu_result = emulator.run(vif, static_argnums=(1,))(emulator.seal(X), False)
    np.testing.assert_allclose(spu_result, ref_result, atol=1e-2)

    # Test with standardized data
    X_standardized = standardize(X)
    ref_result = _statsmodels_vif(np.array(X_standardized), standardized=True)

    spu_result = emulator.run(vif, static_argnums=(1,))(
        emulator.seal(X_standardized), True
    )
    np.testing.assert_allclose(spu_result, ref_result, atol=1e-2)


def main(
    cluster_config: str = emulation.CLUSTER_ABY3_3PC,
    mode: emulation.Mode = emulation.Mode.MULTIPROCESS,
    bandwidth: int = 300,
    latency: int = 20,
):
    with emulation.start_emulator(
        cluster_config,
        mode,
        bandwidth,
        latency,
    ) as emulator:
        emul_vif(emulator)


if __name__ == "__main__":
    main()
