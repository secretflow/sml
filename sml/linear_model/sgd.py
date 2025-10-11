# Copyright 2025 Ant Group Co., Ltd.
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

from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp

from sml.utils.fxp_approx import SigType, sigmoid
from sml.utils.utils import sml_reveal


class Penalty(Enum):
    NONE = "none"
    L1 = "l1"  # not supported
    L2 = "l2"


class Strategy(Enum):
    NAIVE_SGD = "naive_sgd"
    POLICY_SGD = "policy_sgd"


class EarlyStoppingMetric(Enum):
    WEIGHT = "weight"


def _compute_2norm(grad, eps=1e-4):
    return jax.lax.rsqrt(jnp.sum(jnp.square(grad)) + eps)


ActivationFunc = Callable[[jax.Array], jax.Array]


@partial(
    jax.jit,
    static_argnames=(
        "activation",
        "total_batch",
        "batch_size",
        "learning_rate",
        "penalty",
        "l2_norm",
        "strategy",
    ),
)
def _update_weights(
    x: jax.Array,
    y: jax.Array,
    w: jax.Array,
    *,
    activation: ActivationFunc,
    total_batch: int,
    batch_size: int,
    learning_rate: float,
    penalty: Penalty,
    l2_norm: float,
    strategy: Strategy,
) -> jax.Array:
    num_feat = x.shape[1]
    assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
    assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
    w = w.reshape((w.shape[0], 1))

    for idx in range(total_batch):
        begin = idx * batch_size
        end = (idx + 1) * batch_size
        if end > x.shape[0]:
            end = x.shape[0]
        this_batch_size = end - begin
        # padding one col for bias in w
        x_slice = jnp.concatenate(
            (x[begin:end, :], jnp.ones((this_batch_size, 1))), axis=1
        )
        y_slice = y[begin:end, :]

        pred = jnp.matmul(x_slice, w)
        pred = activation(pred)

        err = pred - y_slice
        grad = jnp.matmul(jnp.transpose(x_slice), err)

        if strategy == Strategy.POLICY_SGD:
            scale_factor = _compute_2norm(grad)
        else:
            scale_factor = 1

        if penalty == Penalty.L2:
            reg = l2_norm * jnp.concatenate([w[:-1], jnp.zeros((1, 1))], axis=0)
            grad = grad + reg

        step = (learning_rate * scale_factor * grad) / batch_size

        w = w - step

    return w


@partial(jax.jit, static_argnames=("activation"))
def predict(
    X: jax.Array,
    weights: jax.Array,
    activation: ActivationFunc | None = None,
) -> jax.Array:
    """Make predictions using the trained weights.

    Args:
        X: Input features for prediction.
        weights: Trained weights from SGDClassifier/SGDRegressor.
        activation: Activation function, e.g., sigmoid for SGDClassifier.

    Returns:
        Predicted values.
    """
    n_features = X.shape[1]
    assert weights.shape[0] == n_features + 1, f"w shape is mismatch to x={X.shape}"
    assert weights.ndim == 1 or weights.shape[1] == 1, (
        "weights should be a 1D array or a 2D array with one column"
    )
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    bias = weights[-1, 0]
    weights = weights[:-1]

    pred = jnp.matmul(X, weights) + bias
    return activation(pred) if activation else pred


class SGDBase:
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        decay_epoch: int | None = None,
        decay_rate: float | None = None,
        early_stopping_threshold: bool = 0.0,
        early_stopping_metric: EarlyStoppingMetric = EarlyStoppingMetric.WEIGHT,
        strategy: Strategy = Strategy.NAIVE_SGD,
    ):
        penalty = Penalty(penalty)
        assert epochs > 0, f"epochs<{epochs}> should >0"
        assert learning_rate > 0, f"learning_rate<{learning_rate}> should >0"
        assert batch_size > 0, f"batch_size<{batch_size}> should >0"
        assert penalty != Penalty.L1, "not support L1 penalty for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm<{l2_norm}> should >0 if use L2 penalty"
        if decay_epoch is not None:
            assert decay_epoch > 0, f"decay_epoch<{decay_epoch}> should >0"
            assert decay_rate > 0, f"decay_rate<{decay_rate}> should >0"

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._penalty = penalty
        self._l2_norm = l2_norm
        self._decay_epoch = decay_epoch
        self._decay_rate = decay_rate
        self._strategy = strategy
        self._early_stopping_metric = early_stopping_metric
        self._early_stopping_threshold = early_stopping_threshold
        self._early_stopping_needed = early_stopping_threshold > 0

    def _get_learning_rate(self, epoch_idx: int) -> float:
        if self._decay_rate is not None:
            rate = self._decay_rate ** jnp.floor(epoch_idx / self._decay_epoch)
            return self._learning_rate * rate
        else:
            return self._learning_rate

    def _fit(self, X: jax.Array, y: jax.Array, activation: Callable):
        n_samples, n_features = X.shape
        batch_size = min(self._batch_size, n_samples)
        total_batch = int((n_samples + batch_size - 1) / batch_size)

        weights = jnp.zeros((n_features + 1, 1))
        weights_last = weights
        epoch = 0
        need_stop = False

        def cond_fun(carry):
            _weights, _weights_last, epoch, need_stop = carry
            return jnp.logical_and(epoch < self._epochs, jnp.logical_not(need_stop))

        def body_fun(carry):
            weights, weights_last, epoch, need_stop = carry
            weights_last = weights
            learning_rate = self._get_learning_rate(epoch)
            weights = weights = _update_weights(
                X,
                y,
                weights,
                activation=activation,
                total_batch=total_batch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                penalty=self._penalty,
                l2_norm=self._l2_norm,
                strategy=self._strategy,
            )
            if self._early_stopping_needed:
                if self._early_stopping_metric == EarlyStoppingMetric.WEIGHT:
                    test_criteria = jnp.max(jnp.abs(weights - weights_last))
                    stop = test_criteria < self._early_stopping_threshold
                else:
                    raise NotImplementedError(
                        f"early_stopping_metric={self._early_stopping_metric} is not supported"
                    )
                # WARNING: Reveal the need_stop to plaintext here
                need_stop = sml_reveal(stop)
            else:
                need_stop = False
            return (weights, weights_last, epoch + 1, need_stop)

        weights, weights_last, epoch, need_stop = jax.lax.while_loop(
            cond_fun, body_fun, (weights, weights_last, epoch, need_stop)
        )

        self.n_features_ = n_features
        self.weights_ = weights


class SGDClassifier(SGDBase):
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        decay_epoch: int | None = None,
        decay_rate: float | None = None,
        early_stopping_threshold: bool = 0.0,
        early_stopping_metric: EarlyStoppingMetric = EarlyStoppingMetric.WEIGHT,
        sig_type: SigType = SigType.T1,
        strategy: str | Strategy = Strategy.NAIVE_SGD,
    ):
        sig_type = SigType(sig_type)
        strategy = Strategy(strategy)

        if (
            strategy == Strategy.POLICY_SGD
            and decay_epoch is None
            and decay_rate is None
        ):
            # default early stop strategy for policy-sgd
            decay_rate = 0.5
            decay_epoch = 5

        self._sig_type = sig_type

        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            penalty=penalty,
            l2_norm=l2_norm,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_metric=early_stopping_metric,
            strategy=strategy,
        )

    def fit(self, X: jax.Array, y: jax.Array) -> "SGDClassifier":
        activation = lambda pred: sigmoid(pred, self._sig_type)
        self._fit(X, y, activation)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        def _activation(pred: jax.Array) -> jax.Array:
            return sigmoid(pred, self._sig_type)

        return predict(X, self.weights_, _activation)

    def tree_flatten(self):
        children = (self.weights_,)
        aux_data = {
            "epochs": self._epochs,
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "penalty": self._penalty,
            "l2_norm": self._l2_norm,
            "decay_epoch": self._decay_epoch,
            "decay_rate": self._decay_rate,
            "strategy": self._strategy,
            "early_stopping_metric": self._early_stopping_metric,
            "early_stopping_threshold": self._early_stopping_threshold,
            "sig_type": self._sig_type,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (weights,) = children
        obj = cls.__new__(cls)
        for k, v in aux_data.items():
            setattr(obj, f"_{k}", v)
        obj.weights_ = weights
        return obj


class SGDRegressor(SGDBase):
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        decay_epoch: int | None = None,
        decay_rate: float | None = None,
        early_stopping_threshold: bool = 0.0,
        early_stopping_metric: EarlyStoppingMetric = EarlyStoppingMetric.WEIGHT,
    ):
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            penalty=penalty,
            l2_norm=l2_norm,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_metric=early_stopping_metric,
        )

    def fit(self, X: jax.Array, y: jax.Array) -> "SGDRegressor":
        activation = lambda pred: pred
        self._fit(X, y, activation)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        return predict(X, self.weights_)

    def tree_flatten(self):
        children = (self.weights_,)
        aux_data = {
            "epochs": self._epochs,
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "penalty": self._penalty,
            "l2_norm": self._l2_norm,
            "decay_epoch": self._decay_epoch,
            "decay_rate": self._decay_rate,
            "strategy": self._strategy,
            "early_stopping_metric": self._early_stopping_metric,
            "early_stopping_threshold": self._early_stopping_threshold,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (weights,) = children
        obj = cls.__new__(cls)
        for k, v in aux_data.items():
            setattr(obj, f"_{k}", v)
        obj.weights_ = weights
        return obj


jax.tree_util.register_pytree_node_class(SGDClassifier)
jax.tree_util.register_pytree_node_class(SGDRegressor)
