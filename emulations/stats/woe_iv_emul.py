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
import jax.numpy as jnp
import numpy as np

import emulations.utils.emulation as emulation
from sml.preprocessing.preprocessing import KBinsDiscretizer
from sml.stats.woe_iv import woe_iv


# Calculate reference using numpy
def _numpy_woe_iv(
    X_binned: np.ndarray, y: np.ndarray, n_bins: int, positive_value: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate WOE and IV using numpy as comparison implementation"""
    X_binned_np = np.array(X_binned)
    y_np = np.array(y)

    n_features = X_binned_np.shape[1]
    woe_ref = np.zeros((n_features, n_bins))
    iv_ref = np.zeros(n_features)

    for feature_idx in range(n_features):
        x_col = X_binned_np[:, feature_idx]

        pos_counts = np.zeros(n_bins)
        neg_counts = np.zeros(n_bins)

        for bin_idx in range(n_bins):
            mask = x_col == bin_idx
            pos_counts[bin_idx] = np.sum(y_np[mask] == positive_value)
            neg_counts[bin_idx] = np.sum(y_np[mask] != positive_value)

        total_pos = np.sum(pos_counts)
        total_neg = np.sum(neg_counts)

        epsilon = 1e-10

        pos_pct = pos_counts / (total_pos + epsilon)
        neg_pct = neg_counts / (total_neg + epsilon)

        woe_values = np.log((neg_pct + epsilon) / (pos_pct + epsilon))

        total_counts = pos_counts + neg_counts
        woe_values = np.where(total_counts > 0, woe_values, 0.0)
        iv_value = np.sum((neg_pct - pos_pct) * woe_values)

        woe_ref[feature_idx] = woe_values
        iv_ref[feature_idx] = iv_value

    return woe_ref, iv_ref


def emul_woe_iv(emulator: emulation.Emulator):
    def check(spu_result, ref_result):
        np.testing.assert_allclose(spu_result[0], ref_result[0], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(spu_result[1], ref_result[1], rtol=1e-3, atol=1e-3)

    # Test with synthetic data
    seed = 42
    key = jax.random.PRNGKey(seed)

    # Generate features and target
    n_samples, n_features, n_bins = 1000, 5, 10
    X = jax.random.normal(key, (n_samples, n_features))
    key, subkey = jax.random.split(key)
    y = jax.random.bernoulli(subkey, p=0.3, shape=(n_samples,)).astype(jnp.int32)

    # Discretize features
    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy="quantile")
    X_binned = discretizer.fit_transform(X)

    ref_result = _numpy_woe_iv(np.array(X_binned), np.array(y), n_bins)

    # Run SPU computation
    spu_result = emulator.run(woe_iv, static_argnums=(2, 3))(
        emulator.seal(X_binned), emulator.seal(y), n_bins, 1
    )

    check(spu_result, ref_result)


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
        emul_woe_iv(emulator)


if __name__ == "__main__":
    main()
