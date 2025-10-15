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
from sml.preprocessing import MeanEncoder


def emul_mean_encoder(emulator: emulation.Emulator):
    # Test with standardized data
    X = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])  # (4, 2)
    y = jnp.array([1.0, 2.0, 1.0, 2.0])  # (4,)
    n_classes = 4

    def proc(X: jax.Array, y: jax.Array, n_classes: int):
        encoder = MeanEncoder(n_classes)
        result = encoder.fit_transform(X, y)
        table = encoder.table_
        return table, result

    # Run SPU computation
    spu_result = emulator.run(proc, static_argnums=(2,))(
        emulator.seal(X), emulator.seal(y), n_classes
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
        emul_mean_encoder(emulator)


if __name__ == "__main__":
    main()
