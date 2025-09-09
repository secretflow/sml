# Copyright 2024 Ant Group Co., Ltd.
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

import numpy as np
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from sklearn.metrics import r2_score, roc_auc_score

from sml.linear_model.sgd import SGDClassifier, SGDRegressor
from sml.utils.dataset_utils import load_mock_datasets


@pytest.mark.parametrize(
    "penalty,decay_epoch,decay_rate,early_stopping_threshold,strategy",
    [
        ("none", None, None, 0, "naive_sgd"),
        ("l2", 5, 0.5, 0.1, "policy_sgd"),
    ],
)
def test_sgd_classifier(
    penalty: str,
    decay_epoch: int,
    decay_rate: float,
    early_stopping_threshold: float,
    strategy: str,
):
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    def proc(x: np.ndarray, y: np.ndarray):
        model = SGDClassifier(
            epochs=1,
            learning_rate=0.1,
            penalty=penalty,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
            early_stopping_threshold=early_stopping_threshold,
            strategy=strategy,
        )

        y = y.reshape((y.shape[0], 1))

        return model.fit(x, y).predict(x)

    x, y = load_mock_datasets(
        n_samples=50000,
        n_features=100,
        task_type="bi_classification",
        need_split_train_test=False,
    )
    y_pred = spsim.sim_jax(sim, proc)(x, y)
    score = roc_auc_score(y, y_pred)
    print(f"spu_score: {score}")


@pytest.mark.parametrize(
    "penalty,decay_epoch,decay_rate,early_stopping_threshold",
    [
        ("none", None, None, 0),
        ("l2", 5, 0.5, 0.1),
    ],
)
def test_sgd_regressor(
    penalty: str,
    decay_epoch: int,
    decay_rate: float,
    early_stopping_threshold: float,
):
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    def proc(x: np.ndarray, y: np.ndarray):
        model = SGDRegressor(
            epochs=1,
            penalty=penalty,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
            early_stopping_threshold=early_stopping_threshold,
        )

        y = y.reshape((y.shape[0], 1))

        return model.fit(x, y).predict(x)

    x, y = load_mock_datasets(
        n_samples=50000,
        n_features=100,
        task_type="regression",
        need_split_train_test=False,
    )
    y_pred = spsim.sim_jax(sim, proc)(x, y)
    score = r2_score(y, y_pred)
    print(f"spu_score: {score}")
