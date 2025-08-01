# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score as sk_average_precision_score
from sklearn.metrics import brier_score_loss as sk_brier_score_loss

import emulations.utils.emulation as emulation
from sml.metrics.classification.classification import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# TODO: design the enumation framework, just like py.unittest
# all emulation action should begin with `emul_` (for reflection)
def emul_auc(emulator: emulation.Emulator):
    # Create dataset
    row = 10000
    y_true = np.random.randint(0, 2, (row,))
    y_pred = np.random.random((row,))

    # Run
    result = emulator.run(roc_auc_score)(
        *emulator.seal(y_true, y_pred)
    )  # X, y should be two-dimension array
    print(result)


def emul_Classification(emulator: emulation.Emulator):
    def proc(y_true, y_pred, average="binary", labels=None, pos_label=1, transform=1):
        f1 = f1_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            pos_label=pos_label,
            transform=transform,
        )
        precision = precision_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            pos_label=pos_label,
            transform=transform,
        )
        recall = recall_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            pos_label=pos_label,
            transform=transform,
        )
        accuracy = accuracy_score(y_true, y_pred)
        return f1, precision, recall, accuracy

    def sklearn_proc(y_true, y_pred, average="binary", labels=None, pos_label=1):
        f1 = metrics.f1_score(
            y_true, y_pred, average=average, labels=labels, pos_label=pos_label
        )
        precision = metrics.precision_score(
            y_true, y_pred, average=average, labels=labels, pos_label=pos_label
        )
        recall = metrics.recall_score(
            y_true, y_pred, average=average, labels=labels, pos_label=pos_label
        )
        accuracy = metrics.accuracy_score(y_true, y_pred)
        return f1, precision, recall, accuracy

    def check(spu_result, sk_result):
        for pair in zip(spu_result, sk_result):
            np.testing.assert_allclose(pair[0], pair[1], rtol=1, atol=1e-5)

    # Test binary
    y_true = jnp.array([0, 1, 1, 0, 1, 1])
    y_pred = jnp.array([0, 0, 1, 0, 1, 1])
    spu_result = emulator.run(proc, static_argnums=(2, 5))(
        *emulator.seal(y_true, y_pred), "binary", None, 1, False
    )
    sk_result = sklearn_proc(y_true, y_pred)
    check(spu_result, sk_result)

    # Test multiclass
    y_true = jnp.array([0, 1, 1, 0, 2, 1])
    y_pred = jnp.array([0, 0, 1, 0, 2, 1])
    spu_result = emulator.run(proc, static_argnums=(2, 5))(
        *emulator.seal(y_true, y_pred), None, [0, 1, 2], 1, True
    )
    sk_result = sklearn_proc(y_true, y_pred, average=None, labels=[0, 1, 2])
    check(spu_result, sk_result)


def emul_average_precision_score(emulator: emulation.Emulator):
    def procBinary(y_true, y_score, **kwargs):
        sk_res = sk_average_precision_score(y_true, y_score, **kwargs)
        spu_res = emulator.run(average_precision_score)(
            *emulator.seal(y_true, y_score), **kwargs
        )
        return sk_res, spu_res

    def check(res1, res2):
        return np.testing.assert_allclose(res1, res2, rtol=1e-3, atol=1e-3)

    # --- Test binary classification ---
    # 0-1 labels, no tied value
    y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    y_score = jnp.array([0.1, 0.4, 0.35, 0.8], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    # 0-1 labels, with tied value, even length
    y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    y_score = jnp.array([0.4, 0.4, 0.4, 0.25], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    # 0-1 labels, with tied value, odd length
    y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    y_score = jnp.array([0.4, 0.4, 0.4, 0.25, 0.25], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    # customized labels
    y_true = jnp.array([2, 2, 3, 3], dtype=jnp.int32)
    y_score = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    check(*procBinary(y_true, y_score, pos_label=3))
    # larger random dataset
    y_true = jnp.array(np.random.randint(0, 2, 100), dtype=jnp.int32)
    y_score = jnp.array(np.hstack((0, 1, np.random.random(98))), dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    # single label edge case
    y_true = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    y_score = jnp.array([0.4, 0.25, 0.4, 0.25], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    y_true = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    y_score = jnp.array([0.4, 0.25, 0.4, 0.25], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    # zero score edge case
    y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    y_score = jnp.array([0, 0, 0, 0.25, 0.25], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))
    # score > 1 edge case
    y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
    y_score = jnp.array([1.5, 1.5, 1.5, 0.25, 0.25], dtype=jnp.float32)
    check(*procBinary(y_true, y_score))

    # --- Test multiclass classification ---
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    y_score = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.3, 0.3],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.1, 0.2, 0.7],
        ],
        dtype=jnp.float32,
    )
    classes = jnp.unique(y_true)
    # test over three supported average options
    for average in ["macro", "micro", None]:
        sk_res = sk_average_precision_score(y_true, y_score, average=average)
        spu_res = emulator.run(average_precision_score, static_argnums=(3,))(
            *emulator.seal(y_true, y_score), classes, average
        )
        check(sk_res, spu_res)


def emul_brier_score_loss(emulator: emulation.Emulator):
    def proc_binary(y_true, y_proba, **kwargs):
        sk_res = sk_brier_score_loss(y_true, y_proba, **kwargs)
        spu_res = emulator.run(brier_score_loss)(
            *emulator.seal(y_true, y_proba), **kwargs
        )
        return sk_res, spu_res

    def check(res1, res2):
        np.testing.assert_allclose(res1, res2, rtol=1e-3, atol=1e-3)

    # Test 1: basic binary case
    y_true = jnp.array([0, 1, 1, 0, 1], dtype=jnp.int32)
    y_proba = jnp.array([0.1, 0.8, 0.7, 0.4, 0.9], dtype=jnp.float32)
    check(*proc_binary(y_true, y_proba))

    # Test 2: pos_label = 0
    y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    y_proba = jnp.array([0.9, 0.8, 0.2, 0.1], dtype=jnp.float32)
    check(*proc_binary(y_true, y_proba, pos_label=0))

    # Test 3: with sample_weight
    weights = jnp.array([1, 2, 1, 1], dtype=jnp.float32)
    check(*proc_binary(y_true, y_proba, sample_weight=weights, pos_label=0))

    # Test 4: all positives
    y_true = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    y_proba = jnp.array([0.8, 0.7, 0.9, 0.6], dtype=jnp.float32)
    check(*proc_binary(y_true, y_proba))

    # Test 5: edge probas (0 or 1)
    y_true = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    y_proba = jnp.array([0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    check(*proc_binary(y_true, y_proba))


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
        emul_auc(emulator)
        emul_Classification(emulator)
        emul_average_precision_score(emulator)
        emul_brier_score_loss(emulator)


if __name__ == "__main__":
    main()
