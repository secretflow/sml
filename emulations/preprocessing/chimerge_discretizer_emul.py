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

import emulations.utils.emulation as emulation
from sml.preprocessing.chimerge_discretizer import ChiMergeDiscretizer


def emul_chimerge_discretizer(emulator: emulation.Emulator):
    # Test with standardized data
    n_samples, n_features, n_bins = 100, 5, 5
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (n_samples, n_features))
    key, subkey = jax.random.split(key)
    y = jax.random.bernoulli(subkey, p=0.5, shape=(n_samples,)).astype(jnp.int32)

    def proc(X: jax.Array, y: jax.Array, n_bins: int):
        cm = ChiMergeDiscretizer(n_bins, init_bins=20)
        binned = cm.fit_transform(X, y)
        bin_edges = cm.bin_edges_
        return binned, bin_edges

    # Run SPU computation
    spu_result = emulator.run(proc, static_argnums=(2,))(
        emulator.seal(X), emulator.seal(y), n_bins
    )

    print(f"===>\n {spu_result[0]}, \n {spu_result[1]}")


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
        emul_chimerge_discretizer(emulator)


if __name__ == "__main__":
    main()
