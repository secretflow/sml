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
from sml.utils.utils import sml_reveal, sml_drop_cached_var, sml_make_cached_var


class Penalty(Enum):
    NONE = "none"
    L1 = "l1"  # not supported
    L2 = "l2"


class EarlyStoppingMetric(Enum):
    WEIGHT_ABS = "weight_abs"  # absolute change: max(|w - w_last|) < threshold
    WEIGHT_REL = (
        "weight_rel"  # relative change: max(|w - w_last|) / max(|w|) < threshold
    )


ActivationFunc = Callable[[jax.Array], jax.Array]


def _update_w(
    w: jax.Array,
    y: jax.Array,
    x_with_bias: jax.Array,
    *,
    activation: ActivationFunc,
    batch_size: int,
    learning_rate: float,
    penalty: Penalty,
    l2_norm: float,
    enable_spu_cache: bool = False,
    sig_type: SigType | None = None,
) -> jax.Array:
    """Mini-batch gradient update step.

    Args:
        w: Current weights (including bias), shape: (n_features+1, 1).
        y: Target values for this batch.
        x_with_bias: Input features for this batch (already including bias column).
        activation: Activation function.
        batch_size: Batch size.
        learning_rate: Learning rate.
        penalty: Regularization penalty type.
        l2_norm: L2 regularization coefficient.
        enable_spu_cache: Whether to enable SPU beaver cache for activation.
        sig_type: Sigmoid approximation type, used to determine if cache should be applied.

    Returns:
        Updated weights.
    """
    # Forward pass with optional SPU cache
    # Only use cache when enable_spu_cache=True and sig_type is not T1
    use_cache = enable_spu_cache and sig_type is not None and sig_type != SigType.T1
    pred = jnp.matmul(x_with_bias, w)
    if use_cache:
        pred = sml_make_cached_var(pred)
    pred = activation(pred)
    if use_cache:
        pred = sml_drop_cached_var(pred)

    # Compute error and gradient
    err = pred - y
    grad = jnp.matmul(jnp.transpose(x_with_bias), err)

    # Apply L2 penalty (skip bias term)
    if penalty == Penalty.L2:
        reg = l2_norm * jnp.concatenate([w[:-1], jnp.zeros((1, 1))], axis=0)
        grad = grad + reg

    # Update weights
    step = (learning_rate * grad) / batch_size
    w = w - step

    return w


def _sgd_step(
    x_with_bias: jax.Array,
    y: jax.Array,
    w: jax.Array,
    *,
    activation: ActivationFunc,
    batch_size: int,
    learning_rate: float,
    penalty: Penalty,
    l2_norm: float,
    enable_spu_cache: bool = False,
    sig_type: SigType | None = None,
) -> jax.Array:
    """Process all batches in one epoch.

    Args:
        x_with_bias: Input features with bias column appended.
        y: Target values.
        w: Current weights.
        activation: Activation function.
        batch_size: Batch size.
        learning_rate: Learning rate.
        penalty: Regularization penalty type.
        l2_norm: L2 regularization coefficient.
        enable_spu_cache: Whether to enable SPU beaver cache for activation.
        sig_type: Sigmoid approximation type, used to determine if cache should be applied.

    Returns:
        Updated weights.
    """
    n_samples, _ = x_with_bias.shape
    batch_size = min(batch_size, n_samples)
    num_batches = n_samples // batch_size

    # Process batches using Python for loop
    for idx in range(num_batches):
        start = idx * batch_size
        end = start + batch_size

        # For the last batch, include all remaining samples if any
        if idx == num_batches - 1:
            end = n_samples

        x_b = x_with_bias[start:end, :]
        y_b = y[start:end, :]

        w = _update_w(
            w,
            y_b,
            x_b,
            activation=activation,
            batch_size=end - start,
            learning_rate=learning_rate,
            penalty=penalty,
            l2_norm=l2_norm,
            enable_spu_cache=enable_spu_cache,
            sig_type=sig_type,
        )

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
    assert (
        weights.ndim == 1 or weights.shape[1] == 1
    ), "weights should be a 1D array or a 2D array with one column"
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
        early_stopping_threshold: float = 0.0,
        early_stopping_metric: (
            str | EarlyStoppingMetric
        ) = EarlyStoppingMetric.WEIGHT_REL,
        enable_spu_cache: bool = False,
    ):
        penalty = Penalty(penalty)
        early_stopping_metric = EarlyStoppingMetric(early_stopping_metric)

        assert epochs > 0, f"epochs<{epochs}> should >0"
        assert learning_rate > 0, f"learning_rate<{learning_rate}> should >0"
        assert batch_size > 0, f"batch_size<{batch_size}> should >0"
        assert penalty != Penalty.L1, "not support L1 penalty for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm<{l2_norm}> should >0 if use L2 penalty"

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._penalty = penalty
        self._l2_norm = l2_norm
        self._early_stopping_metric = early_stopping_metric
        self._early_stopping_threshold = early_stopping_threshold
        self._early_stopping_needed = early_stopping_threshold > 0
        self._enable_spu_cache = enable_spu_cache

    def _check_early_stopping(
        self, weights: jax.Array, weights_last: jax.Array
    ) -> bool:
        """Check if early stopping condition is met.

        Returns:
            True if training should stop, False otherwise.
        """
        if not self._early_stopping_needed:
            return False

        # NOTE: VERY VERY EXPENSIVE IN SPU
        max_delta = jnp.max(jnp.abs(weights - weights_last))

        if self._early_stopping_metric == EarlyStoppingMetric.WEIGHT_ABS:
            # Absolute change: max(|w - w_last|) < threshold
            metric_value = max_delta
        else:  # EarlyStoppingMetric.WEIGHT_REL
            # Relative change: max(|w - w_last|) / max(|w|) < threshold
            # We must compute the division first, else SPU may underflow.
            metric_value = max_delta / jnp.max(jnp.abs(weights))

        stop = metric_value < self._early_stopping_threshold
        return sml_reveal(stop)  # type: ignore

    def _fit(
        self,
        X: jax.Array,
        y: jax.Array,
        activation: Callable,
        sig_type: SigType | None = None,
    ):
        n_samples, n_features = X.shape
        batch_size = min(self._batch_size, n_samples)

        # Add bias column to X once, before all epochs
        X_with_bias = jnp.concatenate((X, jnp.ones((n_samples, 1))), axis=1)

        if self._enable_spu_cache:
            X_with_bias = sml_make_cached_var(X_with_bias)

        weights = jnp.zeros((n_features + 1, 1))
        weights_last = weights
        need_stop = False

        def cond_fun(carry):
            _, _, epoch, need_stop = carry
            return jnp.logical_and(epoch < self._epochs, jnp.logical_not(need_stop))

        def body_fun(carry):
            weights, weights_last, epoch, _ = carry
            weights_last = weights
            weights = _sgd_step(
                X_with_bias,
                y,
                weights,
                learning_rate=self._learning_rate,
                activation=activation,
                batch_size=batch_size,
                penalty=self._penalty,
                l2_norm=self._l2_norm,
                enable_spu_cache=self._enable_spu_cache,
                sig_type=sig_type,
            )
            need_stop = self._check_early_stopping(weights, weights_last)
            return (weights, weights_last, epoch + 1, need_stop)

        weights, _, _, _ = jax.lax.while_loop(
            cond_fun, body_fun, (weights, weights_last, 0, need_stop)
        )

        if self._enable_spu_cache:
            X_with_bias = sml_drop_cached_var(X_with_bias)

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
        early_stopping_threshold: float = 0.0,
        early_stopping_metric: (
            str | EarlyStoppingMetric
        ) = EarlyStoppingMetric.WEIGHT_REL,
        sig_type: SigType = SigType.T1,
        enable_spu_cache: bool = False,
    ):
        sig_type = SigType(sig_type)

        def _activation(pred: jax.Array) -> jax.Array:
            return sigmoid(pred, sig_type)

        self._activation = _activation
        self._sig_type = sig_type

        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            penalty=penalty,
            l2_norm=l2_norm,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_metric=early_stopping_metric,
            enable_spu_cache=enable_spu_cache,
        )

    def fit(self, X: jax.Array, y: jax.Array) -> "SGDClassifier":
        self._fit(X, y, self._activation, self._sig_type)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        return predict(X, self.weights_, self._activation)

    def tree_flatten(self):
        children = (self.weights_,)
        aux_data = {
            "epochs": self._epochs,
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "penalty": self._penalty,
            "l2_norm": self._l2_norm,
            "early_stopping_metric": self._early_stopping_metric,
            "early_stopping_threshold": self._early_stopping_threshold,
            "sig_type": self._sig_type,
            "enable_spu_cache": self._enable_spu_cache,
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


def _identity(pred: jax.Array) -> jax.Array:
    return pred


class SGDRegressor(SGDBase):
    def __init__(
        self,
        epochs: int,
        learning_rate: float = 0.1,
        batch_size: int = 1024,
        penalty: str | Penalty = Penalty.NONE,
        l2_norm: float = 0.5,
        early_stopping_threshold: float = 0.0,
        early_stopping_metric: (
            str | EarlyStoppingMetric
        ) = EarlyStoppingMetric.WEIGHT_REL,
        enable_spu_cache: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            penalty=penalty,
            l2_norm=l2_norm,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_metric=early_stopping_metric,
            enable_spu_cache=enable_spu_cache,
        )

    def fit(self, X: jax.Array, y: jax.Array) -> "SGDRegressor":
        self._fit(X, y, _identity)
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
            "early_stopping_metric": self._early_stopping_metric,
            "early_stopping_threshold": self._early_stopping_threshold,
            "enable_spu_cache": self._enable_spu_cache,
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
