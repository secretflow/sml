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

import math
from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp

from sml.utils.fxp_approx import SigType, sigmoid


class Penalty(Enum):
    NONE = "None"
    L1 = "l1"  # not supported
    L2 = "l2"


class Strategy(Enum):
    NAIVE_SGD = "naive_sgd"
    POLICY_SGD = "policy_sgd"


@partial(jax.jit, static_argnames=("base", "num_feat"))
def _init_w(base: float, num_feat: int) -> jax.Array:
    # last one is bias
    return jnp.full((num_feat + 1, 1), base, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("eps_norm", "eps_scale"))
def _convergence(
    old_w: jax.Array, cur_w: jax.Array, eps_norm: float, eps_scale: float
) -> bool:
    max_delta = jnp.max(jnp.abs(cur_w - old_w)) * eps_scale
    max_w = jnp.maximum(jnp.max(jnp.abs(cur_w)), jnp.finfo(cur_w.dtype).eps)

    return (max_delta / max_w) < eps_norm


def _compute_2norm(grad, eps=1e-4):
    return jax.lax.rsqrt(jnp.sum(jnp.square(grad)) + eps)


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
    activation: Callable[[jax.Array], jax.Array],
    total_batch: int,
    batch_size: int,
    learning_rate: float,
    penalty: Penalty,
    l2_norm: float,
    strategy: Strategy,
) -> jax.Array:
    assert x.shape[0] >= total_batch * batch_size, "total batch is too large"
    num_feat = x.shape[1]
    assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
    assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
    w = w.reshape((w.shape[0], 1))

    for idx in range(total_batch):
        begin = idx * batch_size
        end = (idx + 1) * batch_size
        # padding one col for bias in w
        x_slice = jnp.concatenate([x[begin:end], jnp.ones((batch_size, 1))], axis=1)
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


class BaseSGD:
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        epsilon: float = 1e-3,
        decay_epoch: int | None = None,
        decay_rate: float | None = None,
        strategy: Strategy = Strategy.NAIVE_SGD,
    ):
        penalty = Penalty(penalty)
        assert epochs > 0, f"epochs<{epochs}> should >0"
        assert learning_rate > 0, f"learning_rate<{learning_rate}> should >0"
        assert batch_size > 0, f"batch_size<{batch_size}> should >0"
        assert epsilon >= 0, f"epsilon<{epsilon}> should >0"
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
        self._epsilon = epsilon
        self._decay_epoch = decay_epoch
        self._decay_rate = decay_rate
        self._strategy = strategy

        if epsilon > 0:
            self._eps_scale = 2 ** math.floor(-math.log2(epsilon))
            self._eps_norm = epsilon * self._eps_scale

    def _get_learning_rate(self, epoch_idx: int) -> float:
        if self._decay_rate is not None:
            rate = self._decay_rate ** math.floor(epoch_idx / self._decay_epoch)
            return self._learning_rate * rate
        else:
            return self._learning_rate

    def _fit(self, X: jax.Array, y: jax.Array, activation: Callable):
        n_samples, n_features = X.shape
        batch_size = min(self._batch_size, n_samples)
        total_batch = int(n_samples / batch_size)

        weights = _init_w(0, n_features)
        for idx in range(self._epochs):
            old_w = weights
            weights = _update_weights(
                X,
                y,
                weights,
                activation=activation,
                total_batch=total_batch,
                batch_size=batch_size,
                learning_rate=self._get_learning_rate(idx),
                penalty=self._penalty,
                l2_norm=self._l2_norm,
                strategy=self._strategy,
            )

            if (
                self._epsilon > 0
                and idx > 0
                and _convergence(old_w, weights, self._eps_norm, self._eps_scale)
            ):
                # early stop
                break

        self.n_features_ = n_features
        self.weights_ = weights

    def _predict(self, X: jax.Array) -> jax.Array:
        n_features = X.shape[1]
        w: jax.Array = self.weights_
        assert w.shape[0] == n_features + 1, f"w shape is mismatch to x={X.shape}"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w.reshape((w.shape[0], 1))

        bias = w[-1, 0]
        w = jnp.resize(w, (n_features, 1))

        pred = jnp.matmul(X, w) + bias
        return pred


class SGDClassifier(BaseSGD):
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        epsilon: float = 1e-3,
        decay_epoch: int | None = None,
        decay_rate: float | None = None,
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
            epsilon=epsilon,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
            strategy=strategy,
        )

    def fit(self, X: jax.Array, y: jax.Array) -> "SGDClassifier":
        activation = lambda pred: sigmoid(pred, self._sig_type)
        self._fit(X, y, activation)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        pred = self._predict(X)
        pred = sigmoid(pred, self._sig_type)
        return pred


class SGDRegressor(BaseSGD):
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        epsilon: float = 1e-3,
        decay_epoch: int | None = None,
        decay_rate: float | None = None,
    ):
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            penalty=penalty,
            l2_norm=l2_norm,
            epsilon=epsilon,
            decay_epoch=decay_epoch,
            decay_rate=decay_rate,
        )

    def fit(self, X: jax.Array, y: jax.Array) -> "SGDRegressor":
        activation = lambda pred: pred
        self._fit(X, y, activation)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        return self._predict(X)
