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
import pytest

from sml.stats import pearsonr
from sml.utils.extmath import standardize


def _numpy_pearsonr(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


@pytest.mark.parametrize("standardized", [True, False])
@pytest.mark.parametrize("n_samples,n_features", [(80, 5), (100, 3)])
def test_pearsonr(standardized: bool, n_samples: int, n_features: int):
    seed = 42
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (n_samples, n_features))
    X_jax = standardize(X) if standardized else X

    R_jax = pearsonr(X_jax, standardized)
    R_np = _numpy_pearsonr(np.asarray(X_jax))
    np.testing.assert_allclose(np.asarray(R_jax), R_np, atol=1e-6)

    # test in spu
    import spu.libspu as libspu
    import spu.utils.simulation as spsim

    def proc(X: jnp.ndarray):
        return pearsonr(X, standardized)

    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64)
    res = spsim.sim_jax(sim, proc)(X_jax)
    np.testing.assert_allclose(res, R_np, atol=1e-3)
