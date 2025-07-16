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

import numpy as np
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from sklearn.decomposition import NMF as SklearnNMF

from sml.decomposition.nmf import NMF


@pytest.fixture(scope="module")
def setup_test_data():
    print(" ========= start test of NMF package ========= \n")
    random_seed = 0
    np.random.seed(random_seed)
    # NMF must use FM128 now, for heavy use of non-linear & matrix operations
    config = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.ABY3,
        field=libspu.FieldType.FM128,
        fxp_fraction_bits=30,
    )
    sim = spsim.Simulator(3, config)

    # generate some dummy test datas
    test_data = np.random.randint(1, 100, (100, 10)) * 1.0
    n_samples, n_features = test_data.shape

    # random matrix should be generated in plaintext.
    n_components = 5
    random_state = np.random.RandomState(random_seed)
    random_A = random_state.standard_normal(size=(n_components, n_features))
    random_B = random_state.standard_normal(size=(n_samples, n_components))

    # test hyper-parameters settings
    l1_ratio = 0.1
    alpha_W = 0.01

    yield (
        sim,
        test_data,
        n_components,
        random_seed,
        l1_ratio,
        alpha_W,
        random_A,
        random_B,
    )

    print(" ========= test of NMF package end ========= \n")


def _nmf_test_main(
    sim,
    test_data,
    n_components,
    random_seed,
    l1_ratio,
    alpha_W,
    random_A,
    random_B,
    plaintext=True,
    mode="uniform",
):
    # uniform means model is fitted by fit_transform method
    # seperate means model is fitted by first fit then transform
    assert mode in ["uniform", "seperate"]

    # must define here, because test may run simultaneously
    model = (
        SklearnNMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
            l1_ratio=l1_ratio,
            solver="mu",  # sml only implement this solver now.
            alpha_W=alpha_W,
        )
        if plaintext
        else NMF(
            n_components=n_components,
            l1_ratio=l1_ratio,
            alpha_W=alpha_W,
            random_matrixA=random_A,
            random_matrixB=random_B,
        )
    )

    def proc(x):
        if mode == "uniform":
            W = model.fit_transform(x)
        else:
            model.fit(x)
            W = model.transform(x)

        H = model.components_
        X_reconstructed = model.inverse_transform(W)
        err = model.reconstruction_err_

        return W, H, X_reconstructed, err

    run_func = proc if plaintext else spsim.sim_jax(sim, proc)

    return run_func(test_data)


def test_nmf_uniform(setup_test_data):
    print("==============  start test of nmf uniform ==============\n")

    W, H, X_reconstructed, err = _nmf_test_main(
        *setup_test_data, plaintext=False, mode="uniform"
    )
    W_sk, H_sk, X_reconstructed_sk, err_sk = _nmf_test_main(
        *setup_test_data, plaintext=True, mode="uniform"
    )

    np.testing.assert_allclose(err, err_sk, rtol=1, atol=1e-1)
    assert np.allclose(W, W_sk, rtol=1, atol=1e-1)
    assert np.allclose(H, H_sk, rtol=1, atol=1e-1)
    assert np.allclose(X_reconstructed, X_reconstructed_sk, rtol=1, atol=1e-1)

    print("==============  nmf uniform test pass  ==============\n")


def test_nmf_seperate(setup_test_data):
    print("==============  start test of nmf seperate ==============\n")

    W, H, X_reconstructed, err = _nmf_test_main(
        *setup_test_data, plaintext=False, mode="seperate"
    )
    W_sk, H_sk, X_reconstructed_sk, err_sk = _nmf_test_main(
        *setup_test_data, plaintext=True, mode="seperate"
    )

    assert np.allclose(err, err_sk, rtol=1, atol=1e-1)
    assert np.allclose(W, W_sk, rtol=1, atol=1e-1)
    assert np.allclose(H, H_sk, rtol=1, atol=1e-1)
    assert np.allclose(X_reconstructed, X_reconstructed_sk, rtol=1, atol=1e-1)

    print("==============  nmf seperate test pass ==============\n")
