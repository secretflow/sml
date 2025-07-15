# Copyright 2023 Ant Group Co., Ltd.
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

import jax.numpy as jnp
import numpy as np

import emulations.utils.emulation as emulation
from sml.utils.extmath import svd

CONFIG_PATH = "emulations/utils/3pc_128.json"


def emul_svd(emulator: emulation.Emulator):
    print("start svd emulation.")
    np.random.seed(0)

    # ONLY test small matrix for usage purpose
    mat1 = jnp.array(np.random.rand(10, 5))
    mat2 = jnp.array(np.random.rand(5, 10))

    def _check_svd_single(mat, power_iter=100):
        print("start svd emulation test, with shape=", mat.shape)

        jax_u, jax_s, jax_vt = jnp.linalg.svd(mat, full_matrices=False)

        mat_spu = emulator.seal(mat)
        u, s, vt = emulator.run(svd, static_argnums=(1,))(mat_spu, power_iter)

        # 1. check svd shape matching(full_matrices=False)
        assert jax_u.shape == u.shape
        assert jax_s.shape == s.shape
        assert jax_vt.shape == vt.shape

        # 2. check singular values equal
        np.testing.assert_allclose(jax_s, s, rtol=0.1, atol=0.1)

        # 3. check U/Vt (maybe with sign flip)
        np.testing.assert_allclose(
            np.dot(jax_u, jax_vt), np.dot(u, vt), rtol=0.1, atol=0.1
        )

    _check_svd_single(mat1)
    _check_svd_single(mat2)
    print("svd emulation pass.")


def main(
    cluster_config: str = CONFIG_PATH,
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
        emul_svd(emulator)


if __name__ == "__main__":
    main()
