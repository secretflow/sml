# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax.numpy as jnp
import jax.random as jr
import numpy as np
import spu.libspu as libspu
import spu.utils.simulation as spsim
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score
from sklearn.metrics import rand_score as sk_rand_score

from sml.metrics.cluster.cluster import adjusted_rand_score, rand_score

key = jr.PRNGKey(42)


def test_cluster():
    sim = spsim.Simulator.simple(2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM128)

    def proc(labels_true, labels_pred, n_classes, n_clusters):
        sk_ri = sk_rand_score(labels_true, labels_pred)
        sk_ari = sk_adjusted_rand_score(labels_true, labels_pred)
        spu_ri = spsim.sim_jax(sim, rand_score, static_argnums=(2, 3))(
            labels_true, labels_pred, n_classes, n_clusters
        )
        spu_ari = spsim.sim_jax(sim, adjusted_rand_score, static_argnums=(2, 3))(
            labels_true, labels_pred, n_classes, n_clusters
        )
        return (sk_ri, sk_ari), (spu_ri, spu_ari)

    def check(spu_result, sk_result):
        for pair in zip(spu_result, sk_result):
            np.testing.assert_allclose(pair[0], pair[1], rtol=1e-3, atol=1e-3)

    # --- Test perfect match ---
    labels_true = jnp.array([0, 0, 1, 1])
    labels_pred = jnp.array([0, 0, 1, 1])
    check(*proc(labels_true, labels_pred, 2, 2))

    # --- Test another perfect match ---
    labels_true = jnp.array([0, 0, 1, 1])
    labels_pred = jnp.array([1, 1, 0, 0])
    check(*proc(labels_true, labels_pred, 2, 2))

    # --- Test total mismatch ---
    labels_true = jnp.array([0, 1, 2])
    labels_pred = jnp.array([0, 0, 0])
    check(*proc(labels_true, labels_pred, 3, 1))

    labels_true = jnp.array([0, 1])
    labels_pred = jnp.array([0, 0])
    check(*proc(labels_true, labels_pred, 2, 1))

    # --- Test partial match ---
    labels_true = jnp.array([0, 0, 1, 2])
    labels_pred = jnp.array([0, 0, 1, 1])
    check(*proc(labels_true, labels_pred, 3, 2))

    # --- Test with more than 2 clusters ---
    labels_true = jnp.array([0, 1, 2, 3])
    labels_pred = jnp.array([1, 2, 3, 0])
    check(*proc(labels_true, labels_pred, 4, 4))

    labels_true = jnp.array([0, 1, 2, 3, 4, 2, 0, 2, 5])
    labels_pred = jnp.array([2, 0, 2, 5, 0, 1, 2, 3, 4])
    check(*proc(labels_true, labels_pred, 6, 6))

    # --- Test scenarios where n_classes and n_clusters exceed the data range ---
    labels_true = jnp.array([0, 1, 2, 3, 4, 2, 0, 2, 5])
    labels_pred = jnp.array([2, 0, 2, 5, 0, 1, 2, 3, 4])
    check(*proc(labels_true, labels_pred, 6, 10))
    check(*proc(labels_true, labels_pred, 8, 6))
    check(*proc(labels_true, labels_pred, 8, 10))

    # --- Large Test Cases (Require FM128)---
    def generate_large_test_case(n_classes, n_clusters, size):
        global key
        key, subkey1 = jr.split(key)
        labels_true = jr.randint(subkey1, shape=(size,), minval=0, maxval=n_classes)
        key, subkey2 = jr.split(key)
        labels_pred = jr.randint(subkey2, shape=(size,), minval=0, maxval=n_clusters)
        return labels_true, labels_pred

    labels_true, labels_pred = generate_large_test_case(20, 20, 500)
    check(*proc(labels_true, labels_pred, 20, 20))

    labels_true, labels_pred = generate_large_test_case(25, 25, 1000)
    check(*proc(labels_true, labels_pred, 25, 25))
