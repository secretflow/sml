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

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import emulations.utils.emulation as emulation
from sml.linear_model.logistic import LogisticRegression


def load_data(emulator: emulation.Emulator, multi_class="binary"):
    # Create dataset
    if multi_class == "binary":
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    else:
        X, y = load_wine(return_X_y=True, as_frame=True)
    scalar = MinMaxScaler(feature_range=(-2, 2))
    cols = X.columns
    X = scalar.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)

    # mark these data to be protected in SPU
    X_spu, y_spu = emulator.seal(
        X.values, y.values.reshape(-1, 1)
    )  # X, y should be two-dimension array
    return X, y, X_spu, y_spu


def proc(
    x,
    y,
    penalty,
    multi_class="binary",
    early_stopping_threshold=0.0,
    epochs=1,
    batch_size=8,
):
    class_labels = [0, 1] if multi_class == "binary" else [0, 1, 2]
    model = LogisticRegression(
        epochs=epochs,
        learning_rate=0.1,
        batch_size=batch_size,
        solver="sgd",
        penalty=penalty,
        sig_type="sr",
        C=1.0,
        l1_ratio=0.5,
        class_weight=None,
        multi_class=multi_class,
        class_labels=class_labels,
        early_stopping_threshold=early_stopping_threshold,
    )

    model = model.fit(x, y)
    prob = model.predict_proba(x)
    pred = model.predict(x)
    return prob, pred, model.actual_epochs


# Test Binary classification
def emul_LogisticRegression(emulator: emulation.Emulator):
    penalty_list = ["l1", "l2", "elasticnet"]
    print(f"penalty_list={penalty_list}")

    X, y, X_spu, y_spu = load_data(emulator, multi_class="binary")
    for i in range(len(penalty_list)):
        penalty = penalty_list[i]
        # Run
        result = emulator.run(proc, static_argnums=(2, 3))(
            X_spu, y_spu, penalty, "binary"
        )
        # print("Predict result prob: ", result[0])
        # print("Predict result label: ", result[1])
        print(f"{penalty} ROC Score: {roc_auc_score(y.values, result[0])}")


# Test Multi classification
def emul_LogisticRegression_multi_classificatio(emulator: emulation.Emulator):
    X, y, X_spu, y_spu = load_data(emulator, multi_class="ovr")
    # Run
    result = emulator.run(proc, static_argnums=(2, 3))(X_spu, y_spu, "l2", "ovr")
    print(
        f"Multi classification OVR ROC Score: {roc_auc_score(y.values, result[0], multi_class='ovr')}"
    )


def emul_LogisticRegression_with_early_stopping(emulator: emulation.Emulator):
    X, y, X_spu, y_spu = load_data(emulator, multi_class="binary")
    # Run
    result = emulator.run(proc, static_argnums=(2, 3, 4, 5, 6))(
        X_spu,
        y_spu,
        "l2",
        "binary",
        0.1,
        100,
        8,
    )

    assert roc_auc_score(y.values, result[0]) > 0.95
    assert result[2] < 100


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
        emul_LogisticRegression(emulator)
        emul_LogisticRegression_multi_classificatio(emulator)
        emul_LogisticRegression_with_early_stopping(emulator)


if __name__ == "__main__":
    main()
