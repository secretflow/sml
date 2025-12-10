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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sml.preprocessing.chimerge_discretizer import (
    ChiMergeDiscretizer,
    _bincounts,
    _chi2,
    _chimerge_step,
    _is_all_bins_valid,
    _quantile_binning,
    chimerge,
)


def _generate_datas(n_samples: int, n_features: int, seed: int = 42):
    """Generate synthetic data for testing ChiMerge discretization algorithm.

    This function creates a dataset with both correlated and independent features
    to better evaluate the ChiMerge algorithm's performance. Half of the features
    are strongly correlated with the target variable y, while the other half are
    independent noise features.

    The correlated features are generated with different distributions for each
    class of y:
    - For y=0: features follow a normal distribution with mean -1
    - For y=1: features follow a normal distribution with mean +1

    This design ensures that the ChiMerge algorithm will produce distinct chi-square
    statistics for correlated vs. independent features, making the test results
    more meaningful and interpretable.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        seed: Random seed for reproducibility

    Returns:
        X: Generated feature matrix of shape (n_samples, n_features)
        y: Generated target vector of shape (n_samples,)
    """
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    y = jax.random.bernoulli(subkey, p=0.5, shape=(n_samples,)).astype(jnp.int32)

    # Generate features with strong correlation to y
    key, subkey = jax.random.split(key)
    # Half of the features are correlated with y
    n_correlated = n_features // 2
    # Generate different feature distributions for y=0 and y=1 samples
    X_correlated_0 = (
        jax.random.normal(subkey, (n_samples, n_correlated)) - 1.0
    )  # mean = -1
    key, subkey = jax.random.split(key)
    X_correlated_1 = (
        jax.random.normal(subkey, (n_samples, n_correlated)) + 1.0
    )  # mean = +1

    # Select appropriate feature values based on y values
    X_correlated = jnp.where(y[:, None], X_correlated_1, X_correlated_0)

    # The other half of features remain independent (noise features)
    key, subkey = jax.random.split(key)
    X_independent = jax.random.normal(subkey, (n_samples, n_features - n_correlated))

    # Combine correlated and independent features
    X = jnp.concatenate([X_correlated, X_independent], axis=1)

    # Shuffle feature order to mix correlated and independent features
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, jnp.arange(n_features))
    X = X[:, perm]

    return X, y


def py_chi2(freqs: np.ndarray) -> float:
    """
    Compute chi-square statistic between two bins (rows) with class frequencies.
    Python version using numpy to avoid JAX warning.

    Args:
        freqs: np.array of shape (2, n_classes), frequency counts for two adjacent bins.

    Returns:
        chi2: scalar, chi-square statistic.
    """
    freqs = np.asarray(freqs)
    RA, RB = freqs[0].sum(), freqs[1].sum()
    N = RA + RB

    if N == 0:
        return 0.0

    C = freqs[0] + freqs[1]  # (n_classes,): column totals

    epsilon = 1e-8

    EA = (RA * C) / N
    EB = (RB * C) / N

    def _safe_divide(numerator, denominator):
        mask = (np.abs(denominator) > epsilon) & np.isfinite(denominator)
        return np.where(mask, numerator / np.maximum(denominator, epsilon), 0.0)

    term_A = _safe_divide((freqs[0] - EA) ** 2, EA)
    term_B = _safe_divide((freqs[1] - EB) ** 2, EB)

    return term_A.sum() + term_B.sum()


def py_chimerge_step(
    bin_edges: np.ndarray, bin_counts: np.ndarray, cur_n_bins: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Args:
        bin_edges: (n_bins + 1,), bin edges for the feature (padded with inf).
        bin_counts: (n_bins, 2), bin counts where bin_counts[:, 0] are
            negative counts and bin_counts[:, 1] are positive counts (padded with zeros).
        cur_n_bins: scalar, current number of valid bins.
    Returns:
        new_bin_edges: (n_bins + 1,), updated bin edges after merging (padded with inf).
        new_bin_counts: (n_bins, 2), updated bin counts after merging (padded with zeros).
        min_chi: scalar, the minimum chi-square value found.
    """
    n_bins = bin_counts.shape[0]

    # Calculate chi-square values for all adjacent pairs
    def calc_chi2(i):
        # Use slicing to handle dynamic indexing
        freqs = bin_counts[i : i + 2, :]
        return py_chi2(freqs)

    # Create indices for all adjacent pairs
    indices = np.arange(n_bins - 1)
    chi_vec = np.array([calc_chi2(i) for i in indices])

    # Additional constraint: only consider indices < cur_n_bins - 1
    # This ensures we only consider valid adjacent pairs within the current number of bins
    valid_adjacent_mask = indices < (cur_n_bins - 1)
    # Set invalid chi-square values to a large number so they won't be selected
    chi_vec = np.where(valid_adjacent_mask, chi_vec, np.inf)

    # Find the minimum chi-square value and its index
    min_chi = np.min(chi_vec)
    min_idx = np.argmin(chi_vec)

    # Merge the bins with minimum chi-square
    # Update bin_counts
    new_bin_counts = bin_counts.copy()
    new_bin_counts[min_idx, :] = bin_counts[min_idx, :] + bin_counts[min_idx + 1, :]

    # Shift the remaining bins forward to fill the gap
    new_bin_counts[min_idx + 1 : -1, :] = new_bin_counts[min_idx + 2 :, :]
    new_bin_counts[-1, :] = 0  # Fill last position with zeros

    # Update bin_edges: remove the edge between the merged bins
    new_bin_edges = bin_edges.copy()
    # Shift edges forward to fill the gap
    last_point = new_bin_edges[-1]
    new_bin_edges[min_idx + 1 : -1] = new_bin_edges[min_idx + 2 :]
    new_bin_edges[-1] = last_point

    return new_bin_edges, new_bin_counts, min_chi


def py_chimerge(
    bin_edges: np.ndarray,
    bin_counts: np.ndarray,
    n_bins: np.ndarray,
    threshold: float,
    min_samples: int,
) -> np.ndarray:
    """
    Perform ChiMerge discretization on all features (numpy version).

    Args:
        bin_edges: (init_bins + 1, n_features), bin edges for each feature.
        bin_counts: (init_bins, 2, n_features), bin counts where bin_counts[:, 0] are
            negative counts and bin_counts[:, 1] are positive counts.
        n_bins: (n_features,), target number of bins for each feature.
        threshold: scalar, chi-square threshold for merging bins.
        min_samples: scalar, minimum number of samples required to perform merging.

    Returns:
        merged_bin_edges: (max_n_bins, n_features), merged bin edges for each feature.
    """
    bin_edges = np.asarray(bin_edges)
    bin_counts = np.asarray(bin_counts)
    n_bins = np.asarray(n_bins)

    init_n_bins = bin_edges.shape[0] - 1
    n_features = bin_edges.shape[1]
    max_n_bins = np.max(n_bins).item()

    # Initialize result array
    merged_bin_edges = np.full((max_n_bins + 1, n_features), np.inf)

    # Process each feature
    for feature_idx in range(n_features):
        # Get data for this feature
        feature_bin_edges = bin_edges[:, feature_idx].copy()
        feature_bin_counts = bin_counts[:, :, feature_idx].copy()
        feature_n_bins = (
            n_bins[feature_idx] if hasattr(n_bins, "__getitem__") else n_bins
        )

        cur_n_bins = init_n_bins

        all_bins_valid = _is_all_bins_valid(feature_bin_counts, cur_n_bins, min_samples)
        hit_threshold = False
        while True:
            need_stop = (cur_n_bins == 1) or (
                cur_n_bins <= feature_n_bins and hit_threshold and all_bins_valid
            )
            if need_stop:
                break

            # Perform merge step
            new_bin_edges, new_bin_counts, min_chi = py_chimerge_step(
                feature_bin_edges, feature_bin_counts, cur_n_bins
            )

            # Update for next iteration
            feature_bin_edges = new_bin_edges
            feature_bin_counts = new_bin_counts
            cur_n_bins -= 1

            all_bins_valid = _is_all_bins_valid(
                feature_bin_counts, cur_n_bins, min_samples
            )
            hit_threshold = min_chi >= threshold

        # Store result for this feature
        merged_bin_edges[:, feature_idx] = feature_bin_edges[: max_n_bins + 1]

    return merged_bin_edges


def test_quantile_binning():
    # Test with a simple dataset
    # Create data where we know the quantiles
    # Feature 1: [0, 1, 2, ..., 9] repeated 10 times = 100 samples
    # Feature 2: [0, 2, 4, ..., 198] = 100 samples
    x = jnp.array([list(range(10)) * 10, list(range(0, 200, 2))]).T

    # Test with 5 bins
    n_bins = 5
    bin_edges = _quantile_binning(x, n_bins=n_bins)

    # Check output shape
    # The function should return (n_bins + 1, n_features)
    assert bin_edges.shape == (n_bins + 1, 2)

    # Check the actual computed values
    # Feature 1: [0, 1, 2, ..., 9] repeated 10 times
    expected_feature_1 = jnp.array([0.0, 1.800001, 3.600002, 5.400002, 7.200005, 9.0])
    np.testing.assert_array_almost_equal(bin_edges[:, 0], expected_feature_1)

    # Feature 2: [0, 2, 4, ..., 198]
    expected_feature_2 = jnp.array([0.0, 39.600002, 79.200005, 118.8, 158.40001, 198.0])
    np.testing.assert_array_almost_equal(bin_edges[:, 1], expected_feature_2)

    # Test with 10 bins
    n_bins = 10
    bin_edges = _quantile_binning(x, n_bins=n_bins)
    assert bin_edges.shape == (n_bins + 1, 2)

    # Check the actual computed values for 10 bins
    expected_feature_1 = jnp.array(
        [0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.0]
    )
    np.testing.assert_array_almost_equal(bin_edges[:, 0], expected_feature_1, decimal=1)


@pytest.mark.parametrize("n_samples,n_features,n_bins", [(100, 5, 10)])
def test_bincounts(n_samples: int, n_features: int, n_bins: int):
    X, y = _generate_datas(n_samples, n_features)

    bin_edges = _quantile_binning(X, n_bins=n_bins)
    bin_counts = _bincounts(bin_edges, X, y)

    # Check output shape
    assert bin_counts.shape == (n_bins, 2, n_features)


def test_chi2():
    freqs = jnp.array(
        [
            [30, 10],  # bin A: 30 class 0, 10 class 1
            [25, 15],  # bin B: 25 class 0, 15 class 1
            [0, 0],
        ]
    )
    jax_chi = _chi2(freqs)
    py_chi = py_chi2(np.asarray(freqs))

    np.testing.assert_almost_equal(float(jax_chi), py_chi)


def test_chimerge_step():
    # Create a simple test case for _merge_step
    # We'll create bin edges and bin counts for a feature with 4 bins
    bin_edges = jnp.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, jnp.inf]
    )  # 5 edges for 4 bins + padding

    # Create bin counts for 4 bins with 2 classes (negative and positive)
    # Format: (n_bins, 2) where first column is negative counts, second is positive counts
    bin_counts = jnp.array(
        [
            [10, 5],  # Negative and positive counts for bin 1
            [5, 10],  # Negative and positive counts for bin 2
            [20, 10],  # Negative and positive counts for bin 3
            [15, 25],  # Negative and positive counts for bin 4
            [0, 0],  # Padding for the fifth bin
        ]
    )

    # Current number of valid bins
    cur_n_bins = 4

    # Call _merge_step
    new_bin_edges, new_bin_counts, min_chi = _chimerge_step(
        bin_edges, bin_counts, cur_n_bins
    )

    # Check output shapes
    assert new_bin_edges.shape == bin_edges.shape
    assert new_bin_counts.shape == bin_counts.shape

    py_bin_edges, py_bin_counts, py_min_chi = py_chimerge_step(
        np.asarray(bin_edges), np.asarray(bin_counts), cur_n_bins
    )

    np.testing.assert_allclose(float(min_chi), py_min_chi)
    np.testing.assert_almost_equal(np.asarray(new_bin_edges), py_bin_edges)
    np.testing.assert_almost_equal(np.asarray(new_bin_counts), py_bin_counts)


@pytest.mark.parametrize(
    "n_samples,n_features,init_bins,min_samples",
    [(100, 5, 10, 10)],
)
@pytest.mark.parametrize(
    "n_bins",
    [
        5,  # same n_bins
        [3, 4, 5, 6, 7],  # diff n_bins for 5 features
    ],
)
def test_chimerge(
    n_samples: int,
    n_features: int,
    init_bins: int,
    n_bins: int | list,
    min_samples: int,
    threshold: float = 2.706,
):
    X, y = _generate_datas(n_samples, n_features)

    # Call chimerge function with n_bins as a jax.Array
    if isinstance(n_bins, list):
        n_bins = jnp.array(n_bins)
        max_n_bins = jnp.max(n_bins).item()
        assert n_bins.shape[0] == X.shape[1], f"{n_bins.shape}, {X.shape}"
    else:
        max_n_bins = n_bins
        n_bins = jnp.full(n_features, n_bins)

    # Generate initial bin edges using quantile binning
    init_bin_edges = _quantile_binning(X, init_bins)

    # Compute bin counts
    init_bin_counts = _bincounts(init_bin_edges, X, y)

    jax_bin_edges = chimerge(
        init_bin_edges, init_bin_counts, n_bins, max_n_bins, threshold, min_samples
    )

    py_bin_edges = py_chimerge(
        init_bin_edges, init_bin_counts, n_bins, threshold, min_samples
    )

    expected_shape = (max_n_bins + 1, n_features)
    assert jax_bin_edges.shape == expected_shape
    assert py_bin_edges.shape == expected_shape
    np.testing.assert_allclose(np.asarray(jax_bin_edges), py_bin_edges)


class PyChiMergeDiscretizer:
    def __init__(
        self,
        n_bins: int | list[int],
        init_bins: int = 100,
        chi_threshold: float = 2.706,
        min_samples: int = 5,
        positive_value: int = 1,
    ):
        self.n_bins = n_bins
        self.init_bins = init_bins
        self.chi_threshold = chi_threshold
        self.min_samples = min_samples
        self.positive_value = positive_value
        self.bin_edges_: np.ndarray = None
        self.n_features_: int = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyChiMergeDiscretizer":
        X, y = np.asarray(X), np.asarray(y)
        n_features = X.shape[1]

        # Prepare n_bins parameter
        if isinstance(self.n_bins, int):
            n_bins = np.full(n_features, self.n_bins)
        else:
            n_bins = np.array(self.n_bins)

        # Generate initial bin edges using quantile binning
        init_bin_edges = _quantile_binning(X, self.init_bins)
        init_bin_counts = _bincounts(init_bin_edges, X, y, self.positive_value)

        bin_edges = py_chimerge(
            init_bin_edges,
            init_bin_counts,
            n_bins,
            self.chi_threshold,
            self.min_samples,
        )
        self.bin_edges_ = bin_edges
        self.n_features_ = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # Get the number of bins from bin_edges_
        n_bins = self.bin_edges_.shape[0] - 1

        # Compute bin indices for all features at once using broadcasting
        # X shape: (n_samples, n_features)
        # self.bin_edges_ shape: (n_bins+1, n_features)
        # After broadcasting: (n_samples, n_features, n_bins+1)
        bin_indices = (X[..., None] >= self.bin_edges_.T[None, ...]).sum(axis=-1) - 1

        # Clip indices to ensure they are within valid range [0, n_bins-1]
        # Values < 0 go to bin 0, values >= n_bins go to bin n_bins-1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        return bin_indices.astype(int)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform discretized data back to original numerical values.

        This method converts bin indices back to numerical values by using
        the midpoint of each bin as the representative value.

        Parameters
        ----------
        X : np.ndarray
            Discretized data of shape (n_samples, n_features) containing bin indices

        Returns
        -------
        X_original : np.ndarray
            Reconstructed numerical values of shape (n_samples, n_features)
        """
        X = np.asarray(X)

        # Calculate bin midpoints for all features
        # For each bin, midpoint = (left_edge + right_edge) / 2
        left_edges = self.bin_edges_[:-1, :]  # (n_bins, n_features)
        right_edges = self.bin_edges_[1:, :]  # (n_bins, n_features)
        bin_midpoints = (left_edges + right_edges) / 2  # (n_bins, n_features)

        # Clip input bin indices to valid range [0, n_bins-1]
        n_bins = self.bin_edges_.shape[0] - 1
        clipped_indices = np.clip(X, 0, n_bins - 1).astype(int)

        # Use broadcasting to select the appropriate midpoints
        n_samples, n_features = X.shape
        reconstructed = np.zeros((n_samples, n_features))

        for j in range(n_features):
            reconstructed[:, j] = bin_midpoints[clipped_indices[:, j], j]

        return reconstructed


@pytest.mark.parametrize(
    "n_samples,n_features,n_bins",
    [(1000, 5, 10)],
)
def test_chimerge_discretizer(
    n_samples: int,
    n_features: int,
    n_bins: int,
    init_bins: int = 10,
    min_samples: int = 5,
):
    X, y = _generate_datas(n_samples, n_features)

    py_cm = PyChiMergeDiscretizer(n_bins, init_bins, min_samples=min_samples)
    py_binned = py_cm.fit_transform(X, y)
    py_bin_inverse = py_cm.inverse_transform(py_binned)
    py_bin_edges = py_cm.bin_edges_

    # Test the jax version
    jax_cm = ChiMergeDiscretizer(n_bins, init_bins=init_bins, min_samples=min_samples)
    jax_binned = jax_cm.fit_transform(X, y)
    jax_bin_inverse = jax_cm.inverse_transform(jax_binned)
    jax_bin_edges = jax_cm.bin_edges_

    np.testing.assert_allclose(np.asarray(jax_bin_edges), py_bin_edges)
    np.testing.assert_allclose(np.asarray(jax_binned), py_binned)
    np.testing.assert_allclose(np.asarray(jax_bin_inverse), py_bin_inverse)

    def proc(X: jax.Array, y: jax.Array, n_bins: int):
        cm = ChiMergeDiscretizer(n_bins, init_bins=init_bins, min_samples=min_samples)
        binned = cm.fit_transform(X, y)
        bin_inverse = cm.inverse_transform(binned)
        bin_edges = cm.bin_edges_
        return binned, bin_edges, bin_inverse

    import spu.libspu as libspu
    import spu.utils.simulation as spsim

    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64)
    spu_binned, spu_bin_edges, spu_bin_inverse = spsim.sim_jax(
        sim, proc, static_argnums=(2,)
    )(X, y, n_bins)

    np.testing.assert_allclose(
        np.asarray(spu_bin_edges), py_bin_edges, rtol=1e-3, atol=1e-3
    )
    # TODO: SPU atol=3 error is too large, need to further investigate the cause
    np.testing.assert_allclose(
        np.asarray(spu_bin_inverse), py_bin_inverse, rtol=0.1, atol=3
    )
    mismatch_count = np.sum(~np.equal(np.asarray(spu_binned), py_binned))
    total_elements = py_binned.size
    mismatch_ratio = mismatch_count / total_elements
    max_mismatch_ratio = 0.02

    assert (
        mismatch_ratio <= max_mismatch_ratio
    ), f"Too many mismatches: {mismatch_ratio:.2%} > {max_mismatch_ratio:.2%} ({mismatch_count}/{total_elements})"
