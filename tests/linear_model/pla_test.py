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
import pandas as pd
import sklearn.linear_model as sk
import spu.libspu as libspu  # type: ignore
import spu.utils.simulation as spsim
from sklearn.datasets import load_iris

from sml.linear_model.pla import Perceptron


def load_iris_binary():
    """Load iris dataset for binary classification (class 0 vs 1)."""
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal length", "sepal width", "petal length", "petal width"],
    )
    df["label"] = iris.target

    # only use sepal length and sepal width features
    # 100 samples
    data = jnp.array(df.iloc[0:100, [0, 1, -1]])
    x, y = data[:, :-1], data[:, -1]

    # y is -1 or 1
    y = jnp.sign(y)
    y = jnp.where(y <= 0, -1, y)
    y = y.reshape((y.shape[0], 1))

    return x, y


def test_pla():
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    def proc(x, y):
        model = Perceptron(
            max_iter=20,
            eta0=1.0,
            penalty="elasticnet",
            alpha=0.001,
            fit_intercept=True,
            l1_ratio=0.7,
            patience=10,
            batch_size=64,
            early_stopping=True,
            early_stopping_metric="hinge_loss",
        )

        return model.fit(x, y).predict(x)

    # load mock data
    x, y = load_iris_binary()
    n_samples = len(y)

    # compare with sklearn
    sk_pla = sk.Perceptron(
        max_iter=20,
        eta0=1.0,
        penalty="elasticnet",
        alpha=0.001,
        l1_ratio=0.7,
        fit_intercept=True,
    )
    result_sk = sk_pla.fit(x, y).predict(x)
    result_sk = result_sk.reshape(result_sk.shape[0], 1)
    acc_sk = jnp.sum(result_sk == y) / n_samples * 100

    # run with spu
    result = spsim.sim_jax(sim, proc)(x, y)
    result = result.reshape(result.shape[0], 1)
    acc_ = jnp.sum(result == y) / n_samples * 100

    # print acc
    print(f"Accuracy in SKlearn: {acc_sk:.2f}%")
    print(f"Accuracy in SPU: {acc_:.2f}%")


def test_pla_weight_early_stopping():
    """Test PLA with weight-based early stopping in SPU."""
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    def proc(x, y):
        model = Perceptron(
            max_iter=100,
            eta0=1.0,
            penalty="elasticnet",
            alpha=0.001,
            fit_intercept=True,
            l1_ratio=0.7,
            batch_size=64,
            early_stopping=True,
            early_stopping_metric="weight",
            early_stopping_threshold=0.001,
        )
        model.fit(x, y)
        pred = model.predict(x)
        return pred, model.actual_epochs_

    x, y = load_iris_binary()
    n_samples = len(y)

    # Run with SPU
    (result, actual_epochs) = spsim.sim_jax(sim, proc)(x, y)
    result = result.reshape(result.shape[0], 1)
    acc_ = jnp.sum(result == y) / n_samples * 100

    print(f"Weight early stopping - Accuracy: {acc_:.2f}%, Actual epochs: {actual_epochs}")

    # Verify early stopping actually kicked in (should stop before max_iter=100)
    assert actual_epochs < 100, (
        f"Early stopping should have triggered before max_iter, got {actual_epochs} epochs"
    )


def test_pla_no_early_stopping():
    """Test PLA without early stopping runs all max_iter epochs."""
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    def proc(x, y):
        model = Perceptron(
            max_iter=10,
            eta0=1.0,
            penalty="l2",
            alpha=0.001,
            fit_intercept=True,
            batch_size=64,
            early_stopping=False,
        )
        model.fit(x, y)
        return model.predict(x), model.actual_epochs_

    x, y = load_iris_binary()

    # Run with SPU
    (result, actual_epochs) = spsim.sim_jax(sim, proc)(x, y)
    assert actual_epochs == 10, (
        f"Without early stopping, should run all epochs, got {actual_epochs}"
    )
    print(f"No early stopping - used {actual_epochs} epochs as expected")


def test_pla_hinge_loss_early_stopping():
    """Test PLA with hinge_loss based early stopping in SPU."""
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

    def proc(x, y):
        model = Perceptron(
            max_iter=100,
            eta0=1.0,
            penalty="elasticnet",
            alpha=0.001,
            fit_intercept=True,
            l1_ratio=0.7,
            batch_size=64,
            early_stopping=True,
            early_stopping_metric="hinge_loss",
            patience=5,
        )
        model.fit(x, y)
        pred = model.predict(x)
        return pred, model.actual_epochs_

    x, y = load_iris_binary()
    n_samples = len(y)

    # Run with SPU
    (result, actual_epochs) = spsim.sim_jax(sim, proc)(x, y)
    result = result.reshape(result.shape[0], 1)
    acc_ = jnp.sum(result == y) / n_samples * 100

    print(f"Hinge loss early stopping - Accuracy: {acc_:.2f}%, Actual epochs: {actual_epochs}")