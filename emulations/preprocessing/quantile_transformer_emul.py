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

import time

import jax.numpy as jnp
import jax.random as random
import numpy as np
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer

import emulations.utils.emulation as emulation
from sml.preprocessing.quantile_transformer import QuantileTransformer


def emul_quantile_transformer(emulator: emulation.Emulator):
    N_QUANTILES = 50
    RANDOM_STATE = 42
    N_SAMPLES = 100
    N_FEATURES = 2

    def proc_quantile_transform(X, n_quantiles, distribution):
        model = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=distribution,
        )
        model.fit(X)
        X_transformed = model.transform(X)
        X_inversed = model.inverse_transform(X_transformed)
        return X_transformed, X_inversed

    def uniform_test(emulator, X_plaintext):
        output_dist = "uniform"
        X_spu = emulator.seal(X_plaintext)

        start_time = time.time()
        X_transformed_spu, X_inversed_spu = emulator.run(
            proc_quantile_transform, static_argnums=(1, 2)
        )(X_spu, N_QUANTILES, output_dist)
        _ = time.time() - start_time

        # Shape checks
        assert X_transformed_spu.shape == X_plaintext.shape
        assert X_inversed_spu.shape == X_plaintext.shape

        # Reference with sklearn (cap n_quantiles by n_samples like sklearn does)
        sklearn_qt = SklearnQuantileTransformer(
            n_quantiles=min(N_QUANTILES, N_SAMPLES),
            output_distribution=output_dist,
            random_state=RANDOM_STATE,
        )
        X_np = np.array(X_plaintext)
        X_transformed_sklearn = sklearn_qt.fit_transform(X_np)
        X_inversed_sklearn = sklearn_qt.inverse_transform(X_transformed_sklearn)

        # Uniform range and non-collapsed distribution
        assert jnp.all(X_transformed_spu >= -1e-4)
        assert jnp.all(X_transformed_spu <= 1.0 + 1e-4)

        # Compare transformed outputs
        np.testing.assert_allclose(
            X_transformed_spu,
            X_transformed_sklearn,
            rtol=1e-3,
            atol=1e-3,
        )

        # Compare inverse outputs to sklearn and to original (looser like unit test)
        np.testing.assert_allclose(
            X_inversed_spu,
            X_inversed_sklearn,
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            X_inversed_spu,
            X_plaintext,
            rtol=0.1,
            atol=0.5,
        )

    # Generate data consistent with unit test
    key = random.PRNGKey(RANDOM_STATE)
    data_key, _ = random.split(key)
    X_plaintext = random.exponential(data_key, (N_SAMPLES, N_FEATURES)) * 10
    X_plaintext = X_plaintext.astype(jnp.float32)
    assert not jnp.isnan(X_plaintext).any()

    uniform_test(emulator, X_plaintext)


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
        emul_quantile_transformer(emulator)


if __name__ == "__main__":
    main()
