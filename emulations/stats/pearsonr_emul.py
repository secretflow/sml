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
import numpy as np

import emulations.utils.emulation as emulation
from sml.stats import pearsonr
from sml.utils.extmath import standardize


def emul_pearsonr(emulator: emulation.Emulator):
    # Test with standardized data
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (100, 5))
    X_standardized = standardize(X)

    # Calculate reference using numpy
    ref_result = np.corrcoef(X_standardized, rowvar=False)

    # Run SPU computation
    spu_result = emulator.run(pearsonr, static_argnums=(1,))(
        emulator.seal(X_standardized), True
    )

    np.testing.assert_allclose(spu_result, ref_result, rtol=1e-3, atol=1e-3)

    # Test with non-standardized data
    spu_result = emulator.run(pearsonr, static_argnums=(1,))(emulator.seal(X), False)
    ref_result = np.corrcoef(X, rowvar=False)
    np.testing.assert_allclose(spu_result, ref_result, rtol=1e-3, atol=1e-3)


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
        emul_pearsonr(emulator)


if __name__ == "__main__":
    main()
