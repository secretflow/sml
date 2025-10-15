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
import pytest

from sml.preprocessing import KBinsDiscretizer
from sml.preprocessing.encoding.woe_encoder import WoEEncoder, fit_woe, transform_woe


def _py_fit_woe(
    X: np.ndarray, y: np.ndarray, n_bins: int, positive_value: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate WOE and IV using numpy as comparison implementation"""
    X_binned_np = np.array(X)
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


def _py_transform_woe(X: np.ndarray, woe: np.ndarray):
    """Transform X using WOE lookup table with numpy"""
    # X shape: (n_samples, n_features)
    # woe shape: (n_features, n_bins)
    n_samples, n_features = X.shape
    result = np.zeros_like(X, dtype=float)

    for feature_idx in range(n_features):
        # Get the WOE table for this feature
        feature_woe = woe[feature_idx]
        # For each sample, look up the corresponding WOE value
        for sample_idx in range(n_samples):
            bin_idx = int(X[sample_idx, feature_idx])
            result[sample_idx, feature_idx] = feature_woe[bin_idx]

    return result


def _py_fit_transform_woe(
    X: np.ndarray, y: np.ndarray, n_groups: int, positive_value: int = 1
):
    X = np.array(X)
    y = np.array(y)
    woes, ivs = _py_fit_woe(X, y, n_groups, positive_value)
    result = _py_transform_woe(X, woes)
    return result, woes, ivs


def _jax_fit_transform_woe(
    X: jnp.ndarray, y: jnp.ndarray, n_groups: int, positive_value: int = 1
):
    woe, iv = fit_woe(X, y, n_groups, positive_value)
    result = transform_woe(X, woe)
    return result, woe, iv


def _make_data(n_samples: int, n_features: int, n_bins: int, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (n_samples, n_features))
    key, subkey = jax.random.split(key)
    y = jax.random.bernoulli(subkey, p=0.5, shape=(n_samples,)).astype(jnp.int32)

    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy="quantile")
    X_binned = discretizer.fit_transform(X)
    return X_binned, y


@pytest.mark.parametrize(
    "n_samples,n_features,n_bins,positive_value", [(80, 5, 10, 1), (100, 3, 10, 0)]
)
def test_fit_transform(n_samples, n_features, n_bins: int, positive_value: int):
    X_binned, y = _make_data(n_samples, n_features, n_bins)
    py_result, py_woe, py_iv = _py_fit_transform_woe(
        X_binned, y, n_bins, positive_value
    )
    jax_result, jax_woe, jax_iv = _jax_fit_transform_woe(
        X_binned, y, n_bins, positive_value
    )

    assert np.allclose(jax_woe, py_woe), (
        f"woe don't match: JAX {jax_woe}, Python {py_woe}"
    )
    assert np.allclose(jax_iv, py_iv), f"iv don't match: JAX {jax_iv}, Python {py_iv}"
    assert np.allclose(jax_result, py_result), (
        f"Result don't match: JAX {jax_result}, Python {py_result}"
    )


def test_woe_encoder(
    n_samples: int = 100,
    n_features: int = 10,
    n_groups: int = 10,
    positive_value: int = 1,
):
    X_binned, y = _make_data(n_samples, n_features, n_groups)

    def proc(X: jnp.ndarray, y: jnp.ndarray):
        encoder = WoEEncoder(n_groups, positive_value=positive_value)
        result = encoder.fit_transform(X, y)
        return (result, encoder.woe_, encoder.iv_)

    import spu.libspu as libspu
    import spu.utils.simulation as spsim

    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64)
    spu_result, spu_woe, spu_iv = spsim.sim_jax(sim, proc)(X_binned, y)
    py_result, py_woe, py_iv = _py_fit_transform_woe(
        np.array(X_binned), np.array(y), n_groups, positive_value
    )

    assert np.allclose(spu_woe, py_woe, rtol=1e-03, atol=1e-03), (
        f"woe don't match: JAX {spu_woe}, Python {py_woe}"
    )
    assert np.allclose(spu_iv, py_iv, rtol=1e-03, atol=1e-03), (
        f"woe don't match: JAX {spu_iv}, Python {py_iv}"
    )
    assert np.allclose(spu_result, py_result, rtol=1e-03, atol=1e-03), (
        f"Result don't match: JAX {spu_result}, Python {py_result}"
    )


if __name__ == "__main__":
    test_fit_transform()
    test_woe_encoder()
