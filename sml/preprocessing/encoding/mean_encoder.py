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


from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("n_classes", "regularization"))
def fit_mean(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_classes: int,
    regularization: float = 1.0,
) -> jnp.ndarray:
    """
    Mean encoding with fixed lookup table output.

    Args:
        X (jnp.ndarray): (N, D), integer categories in [0, n_classes).
        y (jnp.ndarray): (N,) or (N, 1), target.
        regularization (float): Smoothing strength.
        n_classes (int): max Number of possible classes per feature (i.e., |vocab|).

    Returns:
        jnp.ndarray: Encoding table of shape (D, n_classes), dtype=float32.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional with shape (n_samples, n_features)")
    if y.ndim not in [1, 2]:
        raise ValueError("y must be 1-dimensional or 2-dimensional")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    global_mean = jnp.mean(y)

    # one_hot is (n_samples, n_features, n_classes)
    one_hot = jax.nn.one_hot(X, n_classes, dtype=jnp.float32)

    # class_counts is (n_features, n_classes)
    class_counts = jnp.sum(one_hot, axis=0)

    # y_expanded is (n_samples, 1, 1) for broadcasting
    y_expanded = y[:, None]

    # class_sums is (n_features, n_classes)
    class_sums = jnp.sum(one_hot * y_expanded, axis=0)

    regularized_mean = (class_sums + regularization * global_mean) / (
        class_counts + regularization
    )

    max_class_in_data = jnp.max(X)
    class_indices = jnp.arange(n_classes)

    # default_values is (n_classes,)
    default_values = jnp.where(class_indices <= max_class_in_data, global_mean, jnp.nan)

    # Broadcast default_values to (n_features, n_classes)
    default_values_bc = jnp.broadcast_to(default_values, regularized_mean.shape)

    return jnp.where(class_counts > 0, regularized_mean, default_values_bc)


def transform_mean(X: jnp.ndarray, table: jnp.ndarray) -> jnp.ndarray:
    """
    Transform X using lookup table.

    Args:
        X (jnp.ndarray): (N, D), category indices.
        table (jnp.ndarray): (D, num_classes), output from mean_encode.

    Returns:
        jnp.ndarray: (N, D), encoded values.
    """
    D = X.shape[1]
    feature_indices = jnp.broadcast_to(jnp.arange(D), X.shape)
    return table[feature_indices, X]


class MeanEncoder:
    def __init__(self, n_classes: int, regularization: float = 1.0):
        self.n_classes = n_classes
        self.regularization = regularization

    def fit(self, X: jnp.ndarray, y: jnp.ndarray):
        table = fit_mean(X, y, self.n_classes, self.regularization)
        self.table_ = table

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return transform_mean(X, self.table_)

    def fit_transform(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        self.fit(X, y)
        return self.transform(X)
