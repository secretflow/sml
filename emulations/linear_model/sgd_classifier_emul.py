# Copyright 2023 Ant Group Co., Ltd.
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

import emulations.utils.emulation as emulation
from sml.linear_model.sgd_classifier import SGDClassifier
from sml.utils.dataset_utils import load_mock_datasets


def emul_SGDClassifier(emulator: emulation.Emulator):

    def proc(x, y):
        model = SGDClassifier(
            epochs=1,
            learning_rate=0.1,
            batch_size=1024,
            reg_type="logistic",
            penalty="None",
            l2_norm=0.0,
        )

        y = y.reshape((y.shape[0], 1))

        return model.fit(x, y).predict_proba(x)

    # load mock data
    x, y = load_mock_datasets(
        n_samples=50000,
        n_features=100,
        task_type="bi_classification",
        need_split_train_test=False,
    )

    # mark these data to be protected in SPU
    x, y = emulator.seal(x, y)

    # run
    result = emulator.run(proc)(x, y)
    print(result)


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
        emul_SGDClassifier(emulator)


if __name__ == "__main__":
    main()
