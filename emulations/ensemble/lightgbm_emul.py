# Copyright 2024 Ant Group Co., Ltd.
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
import time

import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier

import emulations.utils.emulation as emulation
from sml.ensemble.lightgbm import LightGBMClassifier as sml_lgbm

MAX_DEPTH = 3


def emul_lightgbm(emulator: emulation.Emulator):
    def proc_wrapper(
        n_estimators,
        learning_rate,
        max_depth,
        num_leaves,
        criterion,
        epsilon,
    ):
        lgbm_custom = sml_lgbm(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            criterion=criterion,
            epsilon=epsilon,
        )

        def proc(X, y):
            lgbm_custom_fit = lgbm_custom.fit(X, y, sample_weight=None)
            result = lgbm_custom_fit.predict(X)
            return result

        return proc

    def load_data():
        iris = load_iris()
        iris_data, iris_label = jnp.array(iris.data), jnp.array(iris.target)
        # sorted_features: n_samples * n_features_in
        n_samples, n_features_in = iris_data.shape
        sorted_features = jnp.sort(iris_data, axis=0)
        new_threshold = (sorted_features[:-1, :] + sorted_features[1:, :]) / 2
        new_features = jnp.greater_equal(
            iris_data[:, :], new_threshold[:, jnp.newaxis, :]
        )
        new_features = new_features.transpose([1, 0, 2]).reshape(n_samples, -1)

        X, y = new_features[:, ::3], iris_label[:]
        return X, y

    # load mock data
    X, y = load_data()

    # compare with sklearn
    gbc = GradientBoostingClassifier(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=MAX_DEPTH,
    )
    start = time.time()
    gbc = gbc.fit(X, y)
    score_plain = gbc.score(X, y)
    end = time.time()
    print(f"Running time in SKlearn: {end - start:.2f}s")

    # mark these data to be protected in SPU
    X_spu, y_spu = emulator.seal(X, y)

    # run
    proc = proc_wrapper(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=MAX_DEPTH,
        num_leaves=31,
        criterion="gini",
        epsilon=1e-5,
    )
    start = time.time()
    result = emulator.run(proc)(X_spu, y_spu)
    end = time.time()
    score_encrypted = jnp.mean(result == y)
    print(f"Running time in SPU: {end - start:.2f}s")

    # print acc
    print(f"Accuracy in SKlearn: {score_plain:.2f}")
    print(f"Accuracy in SPU: {score_encrypted:.2f}")


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
        emul_lightgbm(emulator)


if __name__ == "__main__":
    main()
