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


import jax.numpy as jnp
import numpy as np

from sml.preprocessing.encoding.mean_encoder import (
    MeanEncoder,
    fit_mean,
    transform_mean,
)


def py_fit_mean(
    X: np.ndarray, y: np.ndarray, n_classes: int, regularization: float = 1.0
) -> np.ndarray:
    """
    Pure Python implementation of mean encoding with fixed lookup table output.

    Args:
        X (np.ndarray): (N, D), integer categories in [0, n_classes).
        y (np.ndarray): (N,) or (N, 1), target.
        regularization (float): Smoothing strength.
        n_classes (int): Number of possible classes per feature (i.e., |vocab|).

    Returns:
        np.ndarray: Encoding table of shape (D, n_classes), dtype=float32.
    """
    # Ensure y is the right shape
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Compute global mean
    global_mean = np.mean(y)
    max_class_in_data = np.max(X)

    # Infer n_classes from the data if not provided
    if n_classes is None:
        n_classes = int(np.max(X) + 1)

    # Initialize the encoding table
    D = X.shape[1]  # Number of features
    table = np.zeros((D, n_classes), dtype=np.float32)

    # Compute encodings for each feature
    for feature_idx in range(D):
        feature_col = X[:, feature_idx]

        # Compute encodings for each class in this feature
        for class_idx in range(n_classes):
            # Create mask for this class
            mask = feature_col == class_idx

            # Sum of y values for this class
            class_sum = np.sum(y[mask])

            # Count of samples for this class
            class_count = np.sum(mask.astype(np.float32))

            # Compute regularized mean
            if class_count > 0:
                regularized_mean = (class_sum + regularization * global_mean) / (
                    class_count + regularization
                )
                table[feature_idx, class_idx] = regularized_mean
            else:
                if class_idx <= max_class_in_data:
                    table[feature_idx, class_idx] = global_mean
                else:
                    table[feature_idx, class_idx] = np.nan

    return table


def py_transform_mean(X: np.ndarray, table: np.ndarray) -> np.ndarray:
    """
    Pure Python implementation of transform X using lookup table.

    Args:
        X (np.ndarray): (N, D), category indices.
        table (np.ndarray): (D, num_classes), output from mean_encode.

    Returns:
        np.ndarray: (N, D), encoded values.
    """
    N, D = X.shape
    result = np.zeros((N, D), dtype=table.dtype)

    # Transform each sample and feature
    for sample_idx in range(N):
        for feature_idx in range(D):
            class_idx = X[sample_idx, feature_idx]
            result[sample_idx, feature_idx] = table[feature_idx, class_idx]

    return result


def py_fit_transform_mean(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    table = py_fit_mean(X, y, n_classes)
    encoded = py_transform_mean(X, table)
    return table, encoded


def test_mean_fit_transform():
    # Create test data
    X_jax = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])  # (4, 2)
    y_jax = jnp.array([1.0, 2.0, 1.0, 2.0])  # (4,)
    X_np = np.array(X_jax)
    y_np = np.array(y_jax)
    regularization = 1.0
    n_classes = 2

    # Test mean_fit function
    table_jax = fit_mean(X_jax, y_jax, n_classes, regularization)
    table_py = py_fit_mean(X_np, y_np, n_classes, regularization)

    # Compare JAX and Python implementations
    assert np.allclose(table_jax, table_py), (
        f"Tables don't match: JAX {table_jax}, Python {table_py}"
    )

    result_jax = transform_mean(X_jax, table_jax)
    result_py = py_transform_mean(X_np, table_py)
    assert np.allclose(result_jax, result_py), (
        f"Result don't match: JAX {result_jax}, Python {result_py}"
    )


def test_mean_encoder():
    # Create test data
    X = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])  # (4, 2)
    y = jnp.array([1.0, 2.0, 1.0, 2.0])  # (4,)
    n_classes = int(jnp.max(X) + 1)

    def proc(X: jnp.ndarray, y: jnp.ndarray):
        encoder = MeanEncoder(n_classes, regularization=1.0)
        result = encoder.fit_transform(X, y)
        return encoder.table_, result

    import spu.libspu as libspu
    import spu.utils.simulation as spsim

    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64)
    spu_table, spu_result = spsim.sim_jax(sim, proc)(X, y)
    py_table, py_result = py_fit_transform_mean(np.array(X), np.array(y), n_classes)
    assert np.allclose(spu_table, py_table, rtol=1e-03, atol=1e-03), (
        f"Table don't match: JAX {spu_table}, Python {py_table}"
    )
    assert np.allclose(spu_result, py_result, rtol=1e-03, atol=1e-03), (
        f"Result don't match: JAX {spu_result}, Python {py_result}"
    )


def test_n_classes_greater_than_actual():
    """Test case where n_classes is greater than actual number of classes in data"""
    # Create test data with 2 classes (0, 1)
    X_jax = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])  # (4, 2)
    y_jax = jnp.array([1.0, 2.0, 1.0, 2.0])  # (4,)

    # Set n_classes to 4, which is greater than actual classes (0, 1)
    n_classes = 4

    # Test mean_fit function
    table_jax = fit_mean(X_jax, y_jax, n_classes, 1.0)

    # Check that the table has the correct shape
    assert table_jax.shape == (2, n_classes), (
        f"Expected shape (2, {n_classes}), got {table_jax.shape}"
    )

    # Check that the first two columns have valid values (not NaN)
    assert not jnp.isnan(table_jax[0, 0]), "First column should not be NaN"
    assert not jnp.isnan(table_jax[0, 1]), "Second column should not be NaN"
    assert not jnp.isnan(table_jax[1, 0]), "First column should not be NaN"
    assert not jnp.isnan(table_jax[1, 1]), "Second column should not be NaN"

    # Check that the last two columns are NaN
    assert jnp.isnan(table_jax[0, 2]), "Third column should be NaN"
    assert jnp.isnan(table_jax[0, 3]), "Fourth column should be NaN"
    assert jnp.isnan(table_jax[1, 2]), "Third column should be NaN"
    assert jnp.isnan(table_jax[1, 3]), "Fourth column should be NaN"


def test_n_classes_less_than_actual():
    """Test case where n_classes is less than actual number of classes in data"""
    # Create test data with 3 classes (0, 1, 2)
    X_jax = jnp.array([[0, 1], [1, 2], [2, 0], [0, 1]])  # (4, 2)
    y_jax = jnp.array([1.0, 2.0, 3.0, 1.0])  # (4,)

    # Set n_classes to 2, which is less than actual classes (0, 1, 2)
    n_classes = 2

    # Test mean_fit function
    # Note: This should issue a warning, but we're not capturing it in this test
    table_jax = fit_mean(X_jax, y_jax, n_classes, 1.0)

    # Check that the table has the correct shape
    assert table_jax.shape == (2, n_classes), (
        f"Expected shape (2, {n_classes}), got {table_jax.shape}"
    )

    # The function should still work, but classes 2 will be ignored in the computation
    # We're just checking that it doesn't crash


def test_mean_encoder_edge_cases():
    """Test MeanEncoder with edge cases for n_classes"""
    # Test with n_classes greater than actual classes
    X = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])  # (4, 2)
    y = jnp.array([1.0, 2.0, 1.0, 2.0])  # (4,)

    # Test with more n_classes than actual classes
    encoder = MeanEncoder(n_classes=4, regularization=1.0)
    result = encoder.fit_transform(X, y)

    # Check shape
    assert result.shape == X.shape, f"Expected shape {X.shape}, got {result.shape}"

    # Test with fewer n_classes than actual classes
    X2 = jnp.array([[0, 1], [1, 2], [2, 0], [0, 1]])  # (4, 2) with classes 0,1,2
    encoder2 = MeanEncoder(n_classes=2, regularization=1.0)  # Only 2 classes
    result2 = encoder2.fit_transform(X2, y)

    # Check shape
    assert result2.shape == X2.shape, f"Expected shape {X2.shape}, got {result2.shape}"


if __name__ == "__main__":
    test_mean_fit_transform()
    test_mean_encoder()
    test_n_classes_greater_than_actual()
    test_n_classes_less_than_actual()
    test_mean_encoder_edge_cases()
    print("All tests passed!")
