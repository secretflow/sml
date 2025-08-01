# Copyright 2023 Ant Group Co., Ltd.
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
import numpy as np
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from jax import random
from sklearn.decomposition import PCA as SklearnPCA

from sml.decomposition.pca import PCA

np.random.seed(0)


@pytest.fixture(scope="module")
def setup_sim():
    print(" ========= start test of pca package ========= \n")
    sim64 = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
    config128 = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.ABY3,
        field=libspu.FieldType.FM128,
        fxp_fraction_bits=30,
    )
    config128.enable_pphlo_profile = True
    sim128 = spsim.Simulator(3, config128)
    yield sim64, sim128
    print(" ========= end test of pca package ========= \n")


def test_power(setup_sim):
    sim64, _ = setup_sim
    print("start test power method.")

    # Test fit_transform
    def proc_transform(X):
        model = PCA(
            method="power_iteration",
            n_components=2,
            max_power_iter=200,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances
        X_reconstructed = model.inverse_transform(X_transformed)

        return X_transformed, X_variances, X_reconstructed

    # Create a simple dataset
    X = random.normal(random.PRNGKey(0), (10, 20))

    # Run the simulation
    result = spsim.sim_jax(sim64, proc_transform)(X)

    # The transformed data should have 2 dimensions
    assert result[0].shape[1] == 2

    # The mean of the transformed data should be approximately 0
    assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

    X_np = np.array(X)

    # Run fit_transform using sklearn
    sklearn_pca = SklearnPCA(n_components=2)
    X_transformed_sklearn = sklearn_pca.fit_transform(X_np)

    # Compare the transform results(omit sign)
    np.testing.assert_allclose(
        np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=0.1, atol=0.1
    )

    # Compare the variance results
    assert np.allclose(sklearn_pca.explained_variance_, result[1], rtol=0.1, atol=0.1)

    # Run inverse_transform using sklearn
    X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

    # Compare the results
    assert np.allclose(X_reconstructed_sklearn, result[2], atol=0.01, rtol=0.01)


def test_jacobi(setup_sim):
    sim64, _ = setup_sim
    print("start test serial_jacobi method.")

    def proc_transform(X):
        model = PCA(
            method="serial_jacobi_iteration",
            n_components=4,
            max_jacobi_iter=5,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances
        X_reconstructed = model.inverse_transform(X_transformed)
        X_components = model._components

        return X_transformed, X_variances, X_reconstructed, X_components

    # Create a simple dataset
    X = random.normal(random.PRNGKey(0), (10, 20))

    # Run the simulation
    result = spsim.sim_jax(sim64, proc_transform)(X)

    # The transformed data should have 4 dimensions
    assert result[0].shape[1] == 4

    # The mean of the transformed data should be approximately 0
    assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

    X_np = np.array(X)

    # Run fit_transform using sklearn
    sklearn_pca = SklearnPCA(n_components=4)
    X_transformed_sklearn = sklearn_pca.fit_transform(X_np)

    # Compare the transform results(omit sign)
    assert np.allclose(
        np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=0.1, atol=0.1
    )

    # Compare the variance results
    assert np.allclose(sklearn_pca.explained_variance_, result[1], rtol=0.1, atol=0.1)

    # Run inverse_transform using sklearn
    X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

    # Compare the results
    assert np.allclose(X_reconstructed_sklearn, result[2], atol=0.1, rtol=0.1)

    abs_diff = np.abs(np.abs(X_transformed_sklearn) - np.abs(result[0]))
    rel_error = abs_diff / (np.abs(X_transformed_sklearn) + 1e-6)

    print("avg absolute error:\n", np.mean(abs_diff))
    print("avg relative error:\n", np.mean(rel_error))


def test_rsvd(setup_sim):
    _, sim128 = setup_sim
    print("start test rsvd method.")

    # Test fit_transform
    def proc_transform(X, random_matrix):
        model = PCA(
            method="rsvd",
            n_components=n_components,
            n_oversamples=n_oversamples,
            random_matrix=random_matrix,
            scale=[10000000, 10000],
            max_power_iter=100,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances
        X_reconstructed = model.inverse_transform(X_transformed)

        return X_transformed, X_variances, X_reconstructed

    # Create a simple dataset
    # Note:
    # 1. better for large sample data, like (1000, 20)
    # 2. for small data, it may corrupt because the projection will have large error
    X = np.random.normal(size=(50, 20))

    n_components = 1
    n_oversamples = 10

    # Create random_matrix
    random_matrix = np.random.normal(size=(X.shape[1], n_components + n_oversamples))

    # Run the simulation
    result = spsim.sim_jax(sim128, proc_transform)(X, random_matrix)

    # The transformed data should have n_components dimensions
    assert result[0].shape[1] == n_components

    # The mean of the transformed data should be approximately 0
    assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

    X_np = np.array(X)

    # Run fit_transform using sklearn
    sklearn_pca = SklearnPCA(
        n_components=n_components,
        svd_solver="randomized",
        power_iteration_normalizer="QR",
        random_state=0,
    )
    sklearn_pca.fit(X_np)
    X_transformed_sklearn = sklearn_pca.transform(X_np)

    # Compare the transform results(omit sign)
    assert np.allclose(
        np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=1, atol=0.1
    )

    # Compare the variance results
    assert np.allclose(sklearn_pca.explained_variance_, result[1], rtol=1, atol=0.1)

    # Run inverse_transform using sklearn
    X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

    # Compare the results
    assert np.allclose(X_reconstructed_sklearn, result[2], atol=0.1, rtol=1)
