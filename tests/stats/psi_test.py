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

import jax
import numpy as np
import pytest

from sml.preprocessing.preprocessing import KBinsDiscretizer
from sml.stats.psi import psi


def _numpy_psi(actual: np.ndarray, expect: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Simple numpy implementation of PSI for validation, supporting multiple features"""
    n_features = actual.shape[1]
    psi_values = []

    for i in range(n_features):
        # Calculate histograms for each feature
        actual_hist, _ = np.histogram(actual[:, i], bins=bins[:, i])
        expect_hist, _ = np.histogram(expect[:, i], bins=bins[:, i])

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        actual_hist = actual_hist + epsilon
        expect_hist = expect_hist + epsilon

        # Calculate percentages
        actual_pct = actual_hist / np.sum(actual_hist)
        expect_pct = expect_hist / np.sum(expect_hist)

        # Calculate PSI for this feature
        psi_value = np.sum((actual_pct - expect_pct) * np.log(actual_pct / expect_pct))
        psi_values.append(psi_value)

    return np.array(psi_values)


@pytest.mark.parametrize("n_samples,n_features,n_bins", [(80, 5, 10), (100, 3, 10)])
def test_psi(n_samples: int, n_features: int, n_bins: int, seed: int = 42):
    # Generate random data
    key = jax.random.PRNGKey(seed)
    actual = jax.random.normal(key, (n_samples, n_features))
    expect = jax.random.normal(key, (n_samples, n_features))

    # Create discretizer
    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy="quantile")
    discretizer.fit(expect)
    bin_edges = discretizer.bin_edges_

    res_jax = psi(actual, expect, bin_edges)
    res_np = _numpy_psi(np.asarray(actual), np.asarray(expect), np.asarray(bin_edges))
    np.testing.assert_allclose(res_jax, res_np, rtol=1e-3)
