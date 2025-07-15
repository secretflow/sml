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
import time

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

import emulations.utils.emulation as emulation
from sml.manifold.se import SE


def emul_se(emulator: emulation.Emulator):
    def se(sX, num_samples, num_features, k, num_components):
        embedding = SE(
            n_components=num_components,
            n_neighbors=k,
            n_samples=num_samples,
            n_features=num_features,
        )
        X_transformed = embedding.fit_transform(sX)
        return X_transformed

    # Set sample size and dimensions
    num_samples = 20  # Number of samples, se can meet larger num_samples, such as 150
    num_features = 4  # Sample dimension, se can meet larger num_features, such as 12
    k = 6  # Number of nearest neighbors
    num_components = 3  # Dimension after dimensionality reduction

    # Generate random input
    seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    X = jax.random.uniform(
        key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
    )

    sX = emulator.seal(X)
    ans = emulator.run(
        se,
        static_argnums=(
            1,
            2,
            3,
            4,
        ),
    )(sX, num_samples, num_features, k, num_components)

    # sklearn test
    affinity_matrix = kneighbors_graph(
        np.array(X), n_neighbors=k, mode="distance", include_self=False
    )

    # Make the matrix symmetric
    affinity_dense = affinity_matrix.toarray()
    affinity_matrix = 0.5 * (affinity_dense + affinity_dense.T)
    # print(affinity_matrix)
    embedding = spectral_embedding(
        affinity_matrix, n_components=num_components, random_state=None
    )

    # Since the final calculation result is calculated by the eigenvector, the accuracy cannot reach 1e-3
    np.testing.assert_allclose(jnp.abs(embedding), jnp.abs(ans), rtol=0, atol=1e-2)


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
        emul_se(emulator)


if __name__ == "__main__":
    cluster_config = emulation.CLUSTER_ABY3_3PC
    mode = emulation.Mode.MULTIPROCESS
    bandwidth = 300
    latency = 20
    main(cluster_config, mode, bandwidth, latency)
