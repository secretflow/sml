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
import jax.numpy as jnp
import spu.libspu as libspu  # type: ignore
import spu.utils.simulation as spsim
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier

from sml.ensemble.lightgbm import LightGBMClassifier as sml_lgbm

MAX_DEPTH = 3


def test_lightgbm():
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

    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    X, y = load_data()
    n_samples, n_features = X.shape

    # compare with sklearn
    gbc = GradientBoostingClassifier(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=MAX_DEPTH,
    )
    gbc = gbc.fit(X, y)
    score_plain = gbc.score(X, y)

    # run
    proc = proc_wrapper(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        criterion="gini",
        epsilon=1e-5,
    )

    result = spsim.sim_jax(sim, proc)(X, y)
    print(result)
    score_encrypted = jnp.mean(result == y)

    # print acc
    print(f"Accuracy in SKlearn: {score_plain}")
    print(f"Accuracy in SPU: {score_encrypted}")
