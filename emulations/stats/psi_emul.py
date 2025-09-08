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

import emulations.utils.emulation as emulation
from sml.preprocessing.preprocessing import KBinsDiscretizer
from sml.stats.psi import psi


# Calculate reference using numpy
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


def emul_psi(emulator: emulation.Emulator):
    # Test with synthetic data
    seed = 42
    key = jax.random.PRNGKey(seed)

    # Generate actual and expected data
    n_samples, n_features = 1000, 5
    actual = jax.random.normal(key, (n_samples, n_features))
    key, subkey = jax.random.split(key)
    expect = jax.random.normal(subkey, (n_samples, n_features)) + 0.5  # Slight shift

    # Create bins using KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=10, strategy="quantile")
    discretizer.fit(expect)
    bins = discretizer.bin_edges_

    ref_result = _numpy_psi(np.array(actual), np.array(expect), np.array(bins))

    # Run SPU computation
    spu_result = emulator.run(psi)(
        emulator.seal(actual), emulator.seal(expect), emulator.seal(bins)
    )
    np.testing.assert_allclose(spu_result, ref_result, rtol=1e-3, atol=1e-3)


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
        emul_psi(emulator)


if __name__ == "__main__":
    main()
