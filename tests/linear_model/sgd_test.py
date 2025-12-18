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

import jax
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from sklearn.metrics import r2_score, roc_auc_score

from sml.linear_model.sgd import SGDClassifier, SGDRegressor, SigType
from sml.utils.dataset_utils import load_mock_datasets


@pytest.mark.parametrize(
    "penalty,early_stopping_threshold,early_stopping_metric,enable_spu_cache",
    [
        ("none", 0, "weight_rel", False),
        ("l2", 0.1, "weight_rel", False),
        ("none", 0, "weight_rel", True),  # Test with SPU cache enabled
        ("l2", 0.01, "weight_abs", False),  # Test with absolute early stopping
    ],
)
def test_sgd_classifier(
    penalty: str,
    early_stopping_threshold: float,
    early_stopping_metric: str,
    enable_spu_cache: bool,
):
    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64)

    def proc(x: jax.Array, y: jax.Array):
        model = SGDClassifier(
            epochs=1,
            learning_rate=0.1,
            penalty=penalty,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_metric=early_stopping_metric,
            enable_spu_cache=enable_spu_cache,
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
    print(f"spu_score: {score}, enable_spu_cache: {enable_spu_cache}")


@pytest.mark.parametrize(
    "penalty,early_stopping_threshold,early_stopping_metric,enable_spu_cache",
    [
        ("none", 0, "weight_rel", False),
        ("l2", 0.1, "weight_rel", False),
        ("none", 0, "weight_rel", True),  # Test with SPU cache enabled
        ("l2", 0.01, "weight_abs", False),  # Test with absolute early stopping
    ],
)
def test_sgd_regressor(
    penalty: str,
    early_stopping_threshold: float,
    early_stopping_metric: str,
    enable_spu_cache: bool,
):
    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64)

    def proc(x: jax.Array, y: jax.Array):
        model = SGDRegressor(
            epochs=1,
            penalty=penalty,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_metric=early_stopping_metric,
            enable_spu_cache=enable_spu_cache,
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
    print(f"spu_score: {score}, enable_spu_cache: {enable_spu_cache}")


def test_sgd_classifier_with_cache_and_profile():
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K, field=libspu.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.enable_pphlo_profile = True
    sim = spsim.Simulator(2, config)

    model = SGDClassifier(
        epochs=5,
        batch_size=1024,
        learning_rate=0.1,
        penalty="l2",
        sig_type=SigType.T3,
        early_stopping_threshold=0.0,
        early_stopping_metric="weight_rel",
        enable_spu_cache=True,
    )

    def proc(x: jax.Array, y: jax.Array):
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
    print(f"spu_score: {score}, enable_spu_cache: True")


def test_sgd_classifier_without_cache_and_profile():
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K, field=libspu.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.enable_pphlo_profile = True
    sim = spsim.Simulator(2, config)

    model = SGDClassifier(
        epochs=5,
        batch_size=1024,
        learning_rate=0.1,
        penalty="l2",
        sig_type=SigType.T3,
        early_stopping_threshold=0.0,
        early_stopping_metric="weight_rel",
        enable_spu_cache=False,
    )

    def proc(x: jax.Array, y: jax.Array):
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
    print(f"spu_score: {score}, enable_spu_cache: True")
