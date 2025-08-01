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
from sklearn.datasets import load_iris

import emulations.utils.emulation as emulation
from sml.gaussian_process._gpc import GaussianProcessClassifier


def emul_gpc(emulator: emulation.Emulator):
    def proc(x, y, x_pred):
        model = GaussianProcessClassifier(max_iter_predict=10, n_classes=3)
        model.fit(x, y)

        pred = model.predict(x_pred)
        return pred

    # Create dataset
    x, y = load_iris(return_X_y=True)

    idx = list(range(45, 55)) + list(range(100, 105))
    prd_idx = list(range(0, 5)) + list(range(55, 60)) + list(range(110, 115))
    x_pred = x[jnp.array(prd_idx), :]
    y_pred = y[jnp.array(prd_idx)]
    x = x[jnp.array(idx), :]
    y = y[jnp.array(idx)]

    # mark these data to be protected in SPU
    x, y, x_pred = emulator.seal(x, y, x_pred)
    result = emulator.run(proc)(x, y, x_pred)

    print(result)
    print(y_pred)
    print("Accuracy: ", jnp.sum(result == y_pred) / len(y_pred))


def main(
    cluster_config: str = "emulations/gaussian_process/3pc.json",
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
        emul_gpc(emulator)


if __name__ == "__main__":
    main()
