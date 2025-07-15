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

import warnings

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer

from sml.preprocessing.quantile_transformer import QuantileTransformer


@pytest.fixture
def sim():
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.ABY3,
        field=libspu.FieldType.FM128,
        fxp_fraction_bits=30,
    )
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Feature.*")
    return spsim.Simulator(3, config)


def test_uniform_output(sim):
    print("\n--- Test Quantile Transformer (Uniform Output) ---")
    key = random.PRNGKey(42)
    data_key, _ = random.split(key)
    X_plain = random.exponential(data_key, (100, 2)) * 10

    def proc(X):
        transformer = QuantileTransformer(n_quantiles=50, output_distribution="uniform")
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        X_inversed = transformer.inverse_transform(X_transformed)
        return X_transformed, X_inversed

    X_transformed_spu, X_inversed_spu = spsim.sim_jax(sim, proc)(X_plain)
    sklearn_transformer = SklearnQuantileTransformer(
        n_quantiles=50, output_distribution="uniform", random_state=42
    )
    X_transformed_sklearn = sklearn_transformer.fit_transform(np.array(X_plain))
    X_inversed_sklearn = sklearn_transformer.inverse_transform(X_transformed_sklearn)

    assert X_transformed_spu.shape == X_plain.shape
    assert jnp.all(X_transformed_spu >= -1e-4)
    assert jnp.all(X_transformed_spu <= 1.0 + 1e-4)
    np.testing.assert_allclose(
        X_transformed_spu, X_transformed_sklearn, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(X_inversed_spu, X_inversed_sklearn, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(X_inversed_spu, X_plain, rtol=0.1, atol=0.5)
    print("Uniform output test passed.")


def test_constant_column(sim):
    print("\n--- Test Quantile Transformer (Constant Column) ---")
    key = random.PRNGKey(44)
    data_key, _ = random.split(key)
    X_plain = random.normal(data_key, (100, 3)).astype(jnp.float32)
    X_plain = X_plain.at[:, 1].set(5.0)  # No NaN

    def proc(X):
        transformer = QuantileTransformer(n_quantiles=50, output_distribution="uniform")
        X_transformed = transformer.fit_transform(X)
        X_inversed = transformer.inverse_transform(X_transformed)
        return X_transformed, X_inversed

    X_transformed_spu, X_inversed_spu = spsim.sim_jax(sim, proc)(X_plain)
    sklearn_transformer = SklearnQuantileTransformer(n_quantiles=50, random_state=44)
    X_transformed_sklearn = sklearn_transformer.fit_transform(np.array(X_plain))
    X_inversed_sklearn = sklearn_transformer.inverse_transform(X_transformed_sklearn)

    assert jnp.allclose(X_transformed_spu[:, 1], 0.0, atol=1e-3)
    assert jnp.allclose(X_inversed_spu[:, 1], 5.0, atol=1e-4)
    assert jnp.allclose(X_transformed_spu, X_transformed_sklearn, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(X_inversed_spu, X_inversed_sklearn, atol=1e-3, rtol=1e-3)
    print("Constant column test passed.")
