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

import numpy as np
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from jax import random
from sklearn.decomposition import TruncatedSVD as SklearnSVD

from sml.utils.jacobi_svd import jacobi_svd

np.random.seed(0)


@pytest.fixture(scope="module")
def setup_sim():
    print(" ========= start test of jacobi_svd package ========= \n")
    sim64 = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
    config128 = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.ABY3,
        field=libspu.FieldType.FM128,
        fxp_fraction_bits=30,
    )
    sim128 = spsim.Simulator(3, config128)
    yield sim64, sim128
    print(" ========= end test of jacobi_svd package ========= \n")


def test_jacobi_svd(setup_sim):
    sim64, sim128 = setup_sim
    print("start test jacobi svd.")

    # Test fit_transform
    def proc_transform(A, max_iter=100, compute_uv=True):
        U, singular_values, V_T = jacobi_svd(
            A, max_iter=max_iter, compute_uv=compute_uv
        )
        return U, singular_values, V_T

    # Create a random dataset
    A = random.normal(random.PRNGKey(0), (10, 10))
    A = (A + A.T) / 2

    # Run the simulation
    results = spsim.sim_jax(sim128, proc_transform)(A)

    A_np = np.array(A)

    # Run fit_transform using sklearn
    sklearn_svd = SklearnSVD(n_components=min(A_np.shape))
    sklearn_svd.fit(A_np)
    singular_values_sklearn = sklearn_svd.singular_values_
    singular_matrix_sklearn = sklearn_svd.components_

    # Sort Jacobi results[1] (singular values) in descending order
    sorted_indices = np.argsort(results[1])[::-1]  # Get indices for descending order
    sorted_singular_values = results[1][sorted_indices]
    sorted_V_T = results[2][
        sorted_indices, :
    ]  # Adjust V_T to match the sorted singular values

    # Compare the results
    np.testing.assert_allclose(
        singular_values_sklearn, sorted_singular_values, rtol=0.01, atol=0.01
    )
    np.testing.assert_allclose(
        np.abs(singular_matrix_sklearn), np.abs(sorted_V_T), rtol=0.1, atol=0.1
    )
