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
from sml.utils.utils import sml_reveal, sml_make_cached_var


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


def _identity(x: jax.Array) -> jax.Array:
    """Identity activation function for regression."""
    return x


@partial(jax.jit, static_argnames=("activation",))
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
    """Base class for SGD-based models with SPU beaver cache support.
    Args:
        epochs: Number of training epochs. Must be > 0.
        learning_rate: Learning rate for gradient descent. Must be > 0. Default: 0.1.
        batch_size: Mini-batch size for SGD. Must be > 0. Default: 1024.
        penalty: Regularization type. One of "none", "l2". L1 not supported.
            Default: "none".
        l2_norm: L2 regularization strength. Only used when penalty="l2".
            Must be > 0 when used. Default: 0.5.
        early_stopping_threshold: Threshold for early stopping. If > 0, training
            stops when weight change is below this threshold. Default: 0.0 (disabled).
        early_stopping_metric: Metric for early stopping. One of:
            - "weight_abs": max(|w - w_last|) < threshold
            - "weight_rel": max(|w - w_last|) / max(|w|) < threshold
            Default: "weight_rel".
        enable_spu_cache: Enable SPU beaver cache optimization. When True, the
            feature matrix is cached to reduce communication in secure computation.
            Default: False.

    Attributes:
        n_features_: Number of features in training data.
        weights_: Trained model weights (including bias term).
    """

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
        enable_activation_cache: bool = False,
    ):
        """Fit the model using SGD with optional SPU beaver cache.
        1. Pre-split data into batches using jnp.array_split
        2. Cache the full feature matrix at outer level
        3. Use fori_loop for epochs (or while_loop with early stopping)
        4. Pass return values to drop_cached_var
        """
        n_samples, n_features = X.shape

        if n_samples == 0:
            raise ValueError("Number of samples must be > 0")

        batch_size = min(self._batch_size, n_samples)
        n_iters = max(1, n_samples // batch_size)

        # Add bias column to X once, before all epochs
        X_with_bias = jnp.concatenate((X, jnp.ones((n_samples, 1))), axis=1)

        # Cache feature matrix at outer level
        if self._enable_spu_cache:
            X_with_bias = sml_make_cached_var(X_with_bias)

        # Initialize weights
        w = jnp.zeros((n_features + 1, 1))

        # Pre-split data into batches
        xs = jnp.array_split(X_with_bias, n_iters, axis=0)
        ys = jnp.array_split(y, n_iters, axis=0)

        def _one_epoch(w_):
            """Process one epoch of training."""
            for x_batch, y_batch in zip(xs, ys):
                # Forward pass
                dot = jnp.matmul(x_batch, w_)
                if enable_activation_cache:
                    dot = sml_make_cached_var(dot)

                pred = activation(dot)

                # Compute error and gradient
                err = pred - y_batch
                grad = jnp.matmul(jnp.transpose(x_batch), err)

                # Apply L2 penalty (skip bias term)
                if self._penalty == Penalty.L2:
                    reg = self._l2_norm * jnp.concatenate(
                        [w_[:-1], jnp.zeros((1, 1))], axis=0
                    )
                    grad = grad + reg

                # Update weights
                actual_batch_size = x_batch.shape[0]
                step = (self._learning_rate * grad) / actual_batch_size
                w_ = w_ - step

            return w_

        if self._early_stopping_needed:
            # Use while_loop for early stopping support
            w_last = w
            need_stop = False

            def cond_fun(carry):
                _, _, epoch, need_stop = carry
                return jnp.logical_and(epoch < self._epochs, jnp.logical_not(need_stop))

            def while_body_fun(carry):
                w_, w_last_, epoch, _ = carry
                w_last_ = w_
                w_ = _one_epoch(w_)
                need_stop = self._check_early_stopping(w_, w_last_)
                return (w_, w_last_, epoch + 1, need_stop)

            w, _, _, _ = jax.lax.while_loop(
                cond_fun, while_body_fun, (w, w_last, 0, need_stop)
            )
        else:
            # Use fori_loop for better performance when no early stopping
            def fori_body_fun(_, w_):
                return _one_epoch(w_)

            w = jax.lax.fori_loop(0, self._epochs, fori_body_fun, w)

        self.n_features_ = n_features
        self.weights_ = w


class SGDClassifier(SGDBase):
    """SGD Classifier for binary classification with SPU beaver cache support.

    Implements Stochastic Gradient Descent (SGD) optimization for logistic
    regression. Uses sigmoid activation function with configurable approximation
    type for secure computation.

    Args:
        epochs: Number of training epochs. Must be > 0.
        learning_rate: Learning rate for gradient descent. Default: 0.1.
        batch_size: Mini-batch size for SGD. Default: 1024.
        penalty: Regularization type. One of "none", "l2". Default: "none".
        l2_norm: L2 regularization strength. Default: 0.5.
        early_stopping_threshold: Threshold for early stopping. Default: 0.0.
        early_stopping_metric: Metric for early stopping. Default: "weight_rel".
        sig_type: Sigmoid approximation type for secure computation.
            Options: SigType.T1, SigType.T3, SigType.T5. Default: SigType.T1.
        enable_spu_cache: Enable SPU beaver cache optimization. Default: False.

    Example:
        >>> model = SGDClassifier(epochs=10, learning_rate=0.1, enable_spu_cache=True)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

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
        use_act_cache = self._enable_spu_cache and (self._sig_type != SigType.T1)
        self._fit(X, y, self._activation, enable_activation_cache=use_act_cache)
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


class SGDRegressor(SGDBase):
    """SGD Regressor for regression tasks with SPU beaver cache support.

    Implements Stochastic Gradient Descent (SGD) optimization for linear
    regression. Uses identity activation function (no transformation).

    Args:
        epochs: Number of training epochs. Must be > 0.
        learning_rate: Learning rate for gradient descent. Default: 0.1.
        batch_size: Mini-batch size for SGD. Default: 1024.
        penalty: Regularization type. One of "none", "l2". Default: "none".
        l2_norm: L2 regularization strength. Default: 0.5.
        early_stopping_threshold: Threshold for early stopping. Default: 0.0.
        early_stopping_metric: Metric for early stopping. Default: "weight_rel".
        enable_spu_cache: Enable SPU beaver cache optimization. Default: False.

    Example:
        >>> model = SGDRegressor(epochs=10, learning_rate=0.1, enable_spu_cache=True)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

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
        self._fit(X, y, _identity, enable_activation_cache=False)
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
