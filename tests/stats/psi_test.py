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

    def _compute_feature_distribution(
        data: np.ndarray, feature_bins: np.ndarray, n_bins: int
    ) -> np.ndarray:
        """Compute the distribution for a single feature"""
        indices = np.digitize(data, feature_bins, right=True) - 1
        indices = np.clip(indices, 0, n_bins - 1)
        hist = np.bincount(indices, minlength=n_bins)

        # Calculate percentages without adding epsilon here
        pct = hist / np.sum(hist)
        return pct

    n_features = actual.shape[1]
    n_bins = bins.shape[0] - 1
    psi_values = []

    for i in range(n_features):
        # Compute distributions for actual and expect data
        actual_pct = _compute_feature_distribution(actual[:, i], bins[:, i], n_bins)
        expect_pct = _compute_feature_distribution(expect[:, i], bins[:, i], n_bins)

        # Use the same epsilon as JAX implementation
        eps = 1e-8
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)
        expect_pct = np.where(expect_pct == 0, eps, expect_pct)

        # Calculate PSI for this feature
        psi_value = np.sum((actual_pct - expect_pct) * np.log(actual_pct / expect_pct))
        psi_values.append(psi_value)

    return np.array(psi_values)


@pytest.mark.parametrize("n_samples,n_features,n_bins", [(80, 5, 10), (100, 3, 10)])
@pytest.mark.parametrize("offset", [0, 10])
def test_psi(n_samples: int, n_features: int, n_bins: int, offset: int):
    # Generate random data
    seed = 42
    key = jax.random.PRNGKey(seed)
    actual = jax.random.normal(key, (n_samples, n_features))
    expect = jax.random.normal(key, (n_samples, n_features)) + offset

    # Create discretizer
    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy="quantile")
    discretizer.fit(expect)
    bin_edges = discretizer.bin_edges_

    res_jax = psi(actual, expect, bin_edges)
    res_np = _numpy_psi(np.asarray(actual), np.asarray(expect), np.asarray(bin_edges))
    np.testing.assert_allclose(res_jax, res_np, rtol=1e-3)

    # test in spu
    import spu.libspu as libspu
    import spu.utils.simulation as spsim

    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SEMI2K,
        field=libspu.FieldType.FM128,
        fxp_fraction_bits=48,
    )
    sim = spsim.Simulator(2, config)
    res_spu = spsim.sim_jax(sim, psi)(actual, expect, bin_edges)

    np.testing.assert_allclose(res_np, res_spu, atol=1e-3)
