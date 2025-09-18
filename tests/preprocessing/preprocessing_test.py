# Copyright 2023 Ant Group Co., Ltd.
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

import jax.numpy as jnp
import numpy as np
import pytest
import sklearn
import spu.libspu as libspu
import spu.utils.simulation as spsim
from packaging import version
from sklearn import preprocessing

from sml.preprocessing.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    LabelBinarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
)


@pytest.fixture(scope="module")
def sim():
    return spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)


def test_labelbinarizer(sim):
    def labelbinarize(X, Y):
        transformer = LabelBinarizer(neg_label=-2, pos_label=3)
        transformer.fit(X, n_classes=4)
        transformed = transformer.transform(Y)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([1, 2, 4, 6])
    Y = jnp.array([1, 6])

    transformer = preprocessing.LabelBinarizer(neg_label=-2, pos_label=3)
    transformer.fit(X)
    sk_transformed = transformer.transform(Y)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, labelbinarize)(X, Y)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
    np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0)


def test_labelbinarizer_binary(sim):
    def labelbinarize(X):
        transformer = LabelBinarizer()
        transformed = transformer.fit_transform(X, n_classes=2, unique=False)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([1, -1, -1, 1])

    transformer = preprocessing.LabelBinarizer()
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, labelbinarize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
    np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0)


def test_labelbinarizer_unseen(sim):
    def labelbinarize(X, Y):
        transformer = LabelBinarizer()
        transformer.fit(X, n_classes=3)
        return transformer.transform(Y)

    X = jnp.array([2, 4, 5])
    Y = jnp.array([1, 2, 3, 4, 5, 6])

    transformer = preprocessing.LabelBinarizer()
    transformer.fit(X)
    sk_result = transformer.transform(Y)

    spu_result = spsim.sim_jax(sim, labelbinarize)(X, Y)

    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)


def test_binarizer(sim):
    def binarize(X):
        transformer = Binarizer()
        return transformer.transform(X)

    X = jnp.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    transformer = preprocessing.Binarizer()
    sk_result = transformer.transform(X)

    spu_result = spsim.sim_jax(sim, binarize)(X)

    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)


def test_normalizer(sim):
    def normalize_l1(X):
        transformer = Normalizer(norm="l1")
        return transformer.transform(X)

    def normalize_l2(X):
        transformer = Normalizer()
        return transformer.transform(X)

    def normalize_max(X):
        transformer = Normalizer(norm="max")
        return transformer.transform(X)

    X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])

    transformer_l1 = preprocessing.Normalizer(norm="l1")
    sk_result_l1 = transformer_l1.transform(X)
    transformer_l2 = preprocessing.Normalizer()
    sk_result_l2 = transformer_l2.transform(X)
    transformer_max = preprocessing.Normalizer(norm="max")
    sk_result_max = transformer_max.transform(X)

    spu_result_l1 = spsim.sim_jax(sim, normalize_l1)(X)
    spu_result_l2 = spsim.sim_jax(sim, normalize_l2)(X)
    spu_result_max = spsim.sim_jax(sim, normalize_max)(X)

    np.testing.assert_allclose(sk_result_l1, spu_result_l1, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_l2, spu_result_l2, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)


def test_robustscaler(sim):
    X = jnp.array([[-2, 0.5], [-0.5, 1.5], [0, 10.0], [1, 15.0], [5, 20.0]])
    Y = jnp.array([[3, 2]])

    param_combinations = [(True, True), (True, False), (False, True)]

    for with_centering, with_scaling in param_combinations:
        print(f"\nTesting with_centering={with_centering}, with_scaling={with_scaling}")

        def robustscale(X, Y):
            transformer = RobustScaler(
                with_centering=with_centering,
                with_scaling=with_scaling,
                quantile_range=(25.0, 75.0),
            )
            result1 = transformer.fit_transform(X)
            result2 = transformer.transform(Y)
            result1_restore = transformer.inverse_transform(result1)
            result2_restore = transformer.inverse_transform(result2)
            return result1, result2, result1_restore, result2_restore

        # sklearn基准计算
        sk_transformer = preprocessing.RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=(25.0, 75.0),
        )

        sk_result_1 = sk_transformer.fit_transform(X)
        sk_result_2 = sk_transformer.transform(Y)
        sk_restore_1 = sk_transformer.inverse_transform(sk_result_1)
        sk_restore_2 = sk_transformer.inverse_transform(sk_result_2)

        # SPU模拟计算
        spu_result_1, spu_result_2, spu_restore_1, spu_restore_2 = spsim.sim_jax(
            sim, robustscale
        )(X, Y)

        # 正向变换断言
        np.testing.assert_allclose(sk_result_1, spu_result_1, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(sk_result_2, spu_result_2, rtol=1e-4, atol=1e-4)

        # 逆变换断言
        np.testing.assert_allclose(X, spu_restore_1, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(Y, spu_restore_2, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(sk_restore_1, spu_restore_1, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(sk_restore_2, spu_restore_2, rtol=1e-4, atol=1e-4)


def test_minmaxscaler(sim):
    def minmaxscale(X, Y):
        transformer = MinMaxScaler()
        result1 = transformer.fit_transform(X)
        result2 = transformer.transform(Y)
        return result1, result2

    X = jnp.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    Y = jnp.array([[2, 2]])

    transformer = preprocessing.MinMaxScaler()
    sk_result_1 = transformer.fit_transform(X)
    sk_result_2 = transformer.transform(Y)

    spu_result_1, spu_result_2 = spsim.sim_jax(sim, minmaxscale)(X, Y)

    np.testing.assert_allclose(sk_result_1, spu_result_1, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_2, spu_result_2, rtol=0, atol=1e-4)


def test_minmaxscaler_partial_fit(sim):
    def minmaxscale(X):
        transformer = MinMaxScaler()
        for batch in range(50):
            transformer = transformer.partial_fit(X[batch * 2 : batch * 2 + 2])
        result_min = transformer.data_min_
        result_max = transformer.data_max_
        return result_min, result_max

    rng = np.random.RandomState(0)
    n_features = 30
    n_samples = 1000
    offsets = rng.uniform(-1, 1, size=n_features)
    scales = rng.uniform(1, 10, size=n_features)
    X_2d = rng.randn(n_samples, n_features) * scales + offsets
    X = X_2d

    transformer = MinMaxScaler()
    for batch in range(50):
        transformer = transformer.partial_fit(X[batch * 2 : batch * 2 + 2])

    sk_result_min = transformer.data_min_
    sk_result_max = transformer.data_max_

    spu_result_min, spu_result_max = spsim.sim_jax(sim, minmaxscale)(X)

    np.testing.assert_allclose(sk_result_min, spu_result_min, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)


def test_minmaxscaler_zero_variance(sim):
    def minmaxscale(X, X_new):
        transformer = MinMaxScaler()
        transformer.fit(X, zero_variance=True)
        transformed = transformer.transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        transformed_new = transformer.transform(X_new)
        return transformed, inv_transformed, transformed_new

    X = jnp.array([[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]])
    X_new = jnp.array([[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]])

    transformer = preprocessing.MinMaxScaler()
    transformer.fit(X)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    sk_transformed_new = transformer.transform(X_new)

    spu_transformed, spu_inv_transformed, spu_transformed_new = spsim.sim_jax(
        sim, minmaxscale
    )(X, X_new)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        sk_transformed_new, spu_transformed_new, rtol=0, atol=1e-4
    )


def test_maxabsscaler(sim):
    def maxabsscale(X):
        transformer = MaxAbsScaler()
        result = transformer.fit_transform(X)
        return result

    X = jnp.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    transformer = preprocessing.MaxAbsScaler()
    sk_result = transformer.fit_transform(X)

    spu_result = spsim.sim_jax(sim, maxabsscale)(X)

    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)


def test_maxabsscaler_zero_maxabs(sim):
    def maxabsscale(X, X_new):
        transformer = MaxAbsScaler()
        transformer.fit(X, zero_maxabs=True)
        transformed = transformer.transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        transformed_new = transformer.transform(X_new)
        return transformed, inv_transformed, transformed_new

    X = jnp.array(
        [[0.0, 1.0, +0.5], [0.0, 1.0, -0.3], [0.0, 1.0, +1.5], [0.0, 0.0, +0.0]]
    )
    X_new = jnp.array([[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]])

    transformer = preprocessing.MaxAbsScaler()
    transformer.fit(X)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    sk_transformed_new = transformer.transform(X_new)

    spu_transformed, spu_inv_transformed, spu_transformed_new = spsim.sim_jax(
        sim, maxabsscale
    )(X, X_new)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        sk_transformed_new, spu_transformed_new, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_uniform(sim):
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=5, strategy="uniform")
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]])

    transformer = make_sk_kbins_discretizer(5, strategy="uniform")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_uniform_diverse_n_bins(sim):
    def kbinsdiscretize(X, n_bins):
        transformer = KBinsDiscretizer(
            n_bins=max_bins, diverse_n_bins=n_bins, strategy="uniform"
        )
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([3, 5, 5, 5])
    max_bins = int(jnp.max(n_bins))

    transformer = make_sk_kbins_discretizer(n_bins, strategy="uniform")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, n_bins
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_uniform_diverse_n_bins_no_vectorize(sim):
    # When you set vectorize to False, diverse_n_bins should be public.
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(
            n_bins=max_bins,
            diverse_n_bins=np.array([3, 5, 5, 5]),
            strategy="uniform",
        )
        transformed = transformer.fit_transform(X, vectorize=False)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([3, 5, 5, 5])
    max_bins = int(jnp.max(n_bins))

    transformer = make_sk_kbins_discretizer(n_bins, strategy="uniform")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_quantile(sim):
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=5, strategy="quantile")
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]]
    )

    transformer = make_sk_kbins_discretizer(5, strategy="quantile")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def test_kbinsdiscretizer_quantile_diverse_n_bins(sim):
    def kbinsdiscretize(X, n_bins):
        transformer = KBinsDiscretizer(
            n_bins=max_bins, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(X, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([3, 5, 5, 5])
    max_bins = int(jnp.max(n_bins))

    transformer = make_sk_kbins_discretizer(n_bins, strategy="quantile")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, n_bins
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def test_kbinsdiscretizer_quantile_diverse_n_bins2(sim):
    def kbinsdiscretize(X, n_bins):
        transformer = KBinsDiscretizer(
            n_bins=max_bins, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(X, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    n_bins = jnp.array([4, 5, 5, 5])
    max_bins = int(jnp.max(n_bins))

    transformer = make_sk_kbins_discretizer(n_bins=n_bins, strategy="quantile")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, n_bins
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def test_kbinsdiscretizer_quantile_diverse_n_bins_no_vectorize(sim):
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(
            n_bins=max_bins,
            diverse_n_bins=np.array([3, 5, 5, 5]),
            strategy="quantile",
        )
        transformed = transformer.fit_transform(X, vectorize=False, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([3, 5, 5, 5])
    max_bins = int(jnp.max(n_bins))

    transformer = make_sk_kbins_discretizer(n_bins, strategy="quantile")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def test_kbinsdiscretizer_quantile_eliminate(sim):
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=2, strategy="quantile")
        transformed = transformer.fit_transform(X, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [
            [-1.5, 2.0, -3.5, -0.75],
            [-0.5, 3.0, -2.5, 0.0],
            [0.5, 4.0, -1.5, 1.25],
            [0.5, 4.0, -1.5, 1.25],
        ]
    )

    transformer = make_sk_kbins_discretizer(2, strategy="quantile")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


SK_VERSION = version.parse(sklearn.__version__)


def make_sk_kbins_discretizer(
    n_bins: int, strategy: str
) -> preprocessing.KBinsDiscretizer:
    kwargs = {}
    if strategy == "quantile" and SK_VERSION >= version.parse("1.7"):
        # FIXME: Since version 1.7.0, the implementation algorithm has changed,
        # resulting in some individual tests failing
        kwargs["quantile_method"] = "averaged_inverted_cdf"
    return preprocessing.KBinsDiscretizer(
        n_bins, encode="ordinal", strategy=strategy, subsample=None, **kwargs
    )


def test_kbinsdiscretizer_quantile_sample_weight(sim):
    def kbinsdiscretize(X, sample_weight):
        transformer = KBinsDiscretizer(n_bins=2, strategy="quantile")
        transformed = transformer.fit_transform(
            X, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [
            [0.2, 0.2, 0.3, 0.4],
            [0.5, 1.1, 1.2, 1],
            [0.7, 2.12, 2.3, 2.1],
            [1, 2.51, 2.9, 2.6],
        ]
    )
    sample_weight = jnp.array([1, 1, 3, 1])

    transformer = make_sk_kbins_discretizer(2, strategy="quantile")
    transformer.fit(X, sample_weight=sample_weight)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, sample_weight
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins(sim):
    def kbinsdiscretize(X, n_bins, sample_weight):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(
            X, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [
            [0.2, 0.2, 0.3, 0.4],
            [0.5, 1.1, 1.2, 1],
            [0.7, 2.12, 2.3, 2.1],
            [1, 2.51, 2.9, 2.6],
            [1.3, 2.8, 3.1, 2.12],
            [1.9, 2.91, 3.4, 2.99],
        ]
    )
    n_bins = jnp.array([2, 2, 3, 3])
    sample_weight = jnp.array([1, 1, 3, 1, 1, 1])

    transformer = make_sk_kbins_discretizer(n_bins, strategy="quantile")
    transformer.fit(X, sample_weight=sample_weight)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, n_bins, sample_weight
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins2(sim):
    def kbinsdiscretize(X, n_bins, sample_weight):
        transformer = KBinsDiscretizer(
            n_bins=5, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(
            X, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [
            [1.0, 1.2, 1, 1],
            [2, 2, 2.6, 2.1],
            [3.1, 3.11, 3.48, 3.09],
            [4, 4.1, 4.4, 4.6],
            [5, 5.2, 5.88, 5.11],
            [6, 6.4, 6.2, 6.4],
            [7, 7, 7.2, 7],
        ]
    )
    n_bins = jnp.array([2, 3, 4, 5])
    sample_weight = jnp.array([1, 1, 3, 1, 2, 1, 1])

    transformer = make_sk_kbins_discretizer(n_bins, strategy="quantile")
    transformer.fit(X, sample_weight=sample_weight)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, n_bins, sample_weight
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins_no_vectorize(sim):
    def kbinsdiscretize(X, sample_weight):
        transformer = KBinsDiscretizer(
            n_bins=5, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(
            X, vectorize=False, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [
            [1.0, 1.2, 1, 1],
            [2, 2, 2.6, 2.1],
            [3.1, 3.11, 3.48, 3.09],
            [4, 4.1, 4.4, 4.6],
            [5, 5.2, 5.88, 5.11],
            [6, 6.4, 6.2, 6.4],
            [7, 7, 7.2, 7],
        ]
    )
    n_bins = np.array([2, 3, 4, 5])
    sample_weight = jnp.array([1, 1, 3, 1, 2, 1, 1])

    transformer = make_sk_kbins_discretizer(n_bins, strategy="quantile")
    sk_transformed = transformer.fit_transform(X, sample_weight=sample_weight)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(
        X, sample_weight
    )

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-3)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def test_kbinsdiscretizer_kmeans(sim):
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=4, strategy="kmeans")
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [[-4, -4, -4, -4], [-3, -3, -3, -3], [-2, -2, -2, -2], [-1, -1, -1, -1]]
    )

    transformer = make_sk_kbins_discretizer(4, strategy="kmeans")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_kbinsdiscretizer_kmeans_diverse_n_bins_no_vectorize(sim):
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=np.array([2, 3, 3, 3]), strategy="kmeans"
        )
        transformed = transformer.fit_transform(X, vectorize=False, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])

    transformer = make_sk_kbins_discretizer(n_bins, strategy="kmeans")
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)

    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)

    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def test_onehotEncoder(sim):
    manual_categories = [[1, 1.1, 3.25], [2.0, 4.32, 6.10]]

    X = jnp.array([[1, 2.0], [3.25, 4.32], [1.1, 6.10]], dtype=jnp.float64)
    Y = jnp.array([[1, 2.1], [3.21, 4.32], [1.1, 6.10]], dtype=jnp.float64)

    sk_X = np.array([[1, 2.0], [3.25, 4.32], [1.1, 6.10]], dtype=np.float64)
    sk_Y = np.array([[1, 2.1], [3.21, 4.32], [1.1, 6.10]], dtype=np.float64)

    def onehotEncode(X, Y):
        onehotEncoder = OneHotEncoder(categories=manual_categories)
        onehotEncoder.fit(X)
        encoded = onehotEncoder.transform(Y)
        inverse_v = onehotEncoder.inverse_transform(encoded)
        return encoded, inverse_v

    sk_onehotEncoder = preprocessing.OneHotEncoder(
        categories=manual_categories, handle_unknown="ignore", sparse_output=False
    )
    sk_onehotEncoder.fit(sk_X)
    sk_transformed = sk_onehotEncoder.transform(sk_Y)
    sk_inv_transformed = sk_onehotEncoder.inverse_transform(sk_transformed)
    sk_inv_transformed = sk_inv_transformed.astype(np.float64)
    sk_inv_transformed = np.where(np.isnan(sk_inv_transformed), 0.0, sk_inv_transformed)
    spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, onehotEncode)(X, Y)
    spu_inv_transformed = spu_inv_transformed.astype(np.float64)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=1e-3, atol=1e-3
    )
