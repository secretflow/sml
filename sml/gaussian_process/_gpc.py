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

import abc

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.special import erf

from sml.gaussian_process.kernels import RBF, Kernel
from sml.gaussian_process.laplace import _BinaryGaussianProcessClassifierLaplace

LAMBDAS = jnp.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, jnp.newaxis]
COEFS = jnp.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, jnp.newaxis]


def _binary_predict(X: jax.Array, X_train: jax.Array, alpha: jax.Array, kernel: Kernel):
    X = jnp.asarray(X)
    K_star = kernel(X_train, X)
    # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
    f_star = K_star.T.dot(alpha)

    return jnp.where(f_star > 0, 1, 0)


def _binary_predict_proba(
    X: jax.Array,
    X_train: jax.Array,
    alpha: jax.Array,
    W_sr: jax.Array,
    L: jax.Array,
    kernel: Kernel,
):
    K_star = kernel(X_train, X)
    # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
    f_star = K_star.T.dot(alpha)
    v = solve(L, W_sr[:, jnp.newaxis] * K_star)
    # var_f_star = jnp.diag(self.kernel_(X)) - jnp.einsum("ij,ij->j", v, v)
    var_f_star = kernel.diag(X) - jnp.einsum("ij,ij->j", v, v)

    alpha = 1 / (2 * var_f_star)
    gamma = LAMBDAS * f_star

    integrals = (
        jnp.sqrt(jnp.pi / alpha)
        * erf(gamma * jnp.sqrt(alpha / (alpha + LAMBDAS**2)))
        / (2 * jnp.sqrt(var_f_star * 2 * jnp.pi))
    )
    pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

    return jnp.vstack((1 - pi_star, pi_star)).T


def _get_probabilities(
    X: jax.Array,
    X_train: jax.Array,
    alpha: jax.Array,
    W_sr: jax.Array,
    L: jax.Array,
    n_classes: int,
    kernel: Kernel,
) -> jax.Array:
    def single_class_pred(i):
        return _binary_predict_proba(X, X_train, alpha[i], W_sr[i], L[i], kernel)[:, 1]

    return jax.vmap(single_class_pred)(jnp.arange(n_classes))


def predict(
    X: jax.Array,  # (n_test, n_features)
    X_train: jax.Array,  # (n_train, n_features)
    alpha: jax.Array,  # (n_classes, n_train)
    W_sr: jax.Array,  # (n_classes, n_train)
    L: jax.Array,  # (n_classes, n_train, n_train)
    n_classes: int,
    multi_class: str = "one_vs_rest",
    kernel: Kernel | None = None,
) -> jax.Array:
    if kernel is None:
        kernel = RBF()
    if n_classes <= 2:
        return _binary_predict(X, X_train, alpha[0])
    elif multi_class == "one_vs_rest":
        pos_probs = _get_probabilities(X, X_train, alpha, W_sr, L, n_classes, kernel)
        return pos_probs.argmax(axis=0)
    elif multi_class == "one_vs_one":
        raise ValueError("one_vs_one classifier is not supported")
    else:
        raise ValueError("Unknown multi-class mode %s" % multi_class)


def predict_proba(
    X: jax.Array,  # (n_test, n_features)
    X_train: jax.Array,  # (n_train, n_features)
    alpha: jax.Array,  # (n_classes, n_train)
    W_sr: jax.Array,  # (n_classes, n_train)
    L: jax.Array,  # (n_classes, n_train, n_train)
    n_classes: int,
    multi_class: str = "one_vs_rest",
    kernel: Kernel | None = None,
) -> jax.Array:
    if kernel is None:
        kernel = RBF()
    if n_classes <= 2:
        return _binary_predict_proba(X, X_train, alpha[0])
    elif multi_class == "one_vs_rest":
        pos_probs = _get_probabilities(X, X_train, alpha, W_sr, L, n_classes, kernel)
        return pos_probs.T
    elif multi_class == "one_vs_one":
        raise ValueError("one_vs_one classifier is not supported")
    else:
        raise ValueError("Unknown multi-class mode %s" % multi_class)


class Estimator(abc.ABC):
    @property
    @abc.abstractmethod
    def state(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()


class BinaryEstimator(Estimator):
    def __init__(self, **kwargs):
        self.estimator = _BinaryGaussianProcessClassifierLaplace(**kwargs)

    @property
    def state(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        e = self.estimator
        return e.alpha[None, :], e.W_sr_[None, :], e.L_[None, :]

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self


class OneVsRestEstimator(Estimator):
    def __init__(self, n_classes: int, **kwargs):
        self.classes_ = n_classes
        self.estimators_ = [
            _BinaryGaussianProcessClassifierLaplace(**kwargs) for _ in range(n_classes)
        ]

    @property
    def state(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        alpha = []
        W_sr = []
        L = []
        for e in self.estimators_:
            alpha.append(e.alpha)
            W_sr.append(e.W_sr_)
            L.append(e.L_)

        return jnp.stack(alpha), jnp.stack(W_sr), jnp.stack(L)

    def fit(self, X, y):
        for i in range(self.classes_):
            self.estimators_[i].fit(X, jnp.where(y == i, 1, 0))
        return self


class GaussianProcessClassifier:
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    IMPORTANT NOTES:
      1. Current implementations will not optimize the parameters of kernel during training.
      2. ONLY RBF kernel is supported now.
      3. ONLY one_vs_rest mode is supported for multi-class tasks. (You should pre-process your label to 0,1,2... like)
      4. You MUST specify n_classes explicitly, because we can not do data inspections under MPC.

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default.
        Only RBF kernel is supported now.

    max_iter_predict : int, default=20
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    multi_class : 'one_vs_rest', default='one_vs_rest'
        Specifies how multi-class classification problems are handled.
        One binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest.

    poss : "sigmoid", allable or None, default="sigmoid", the predefined
        likelihood function which computes the possibility of the predict output
        w.r.t. f value.

    n_classes : int
        The number of classes in the training data.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    """

    def __init__(
        self,
        kernel=None,
        *,
        poss="sigmoid",
        max_iter_predict=20,
        multi_class="one_vs_rest",
        n_classes=2,
    ):
        self.kernel = kernel
        self.max_iter_predict = max_iter_predict
        self.multi_class = multi_class
        self.poss = poss

        self.n_classes = n_classes

        assert self.n_classes > 1, (
            "GaussianProcessClassifier requires 2 or more distinct classes"
        )

    def tree_flatten(self):
        static_data = (
            self.kernel,
            self.max_iter_predict,
            self.poss,
            self.multi_class,
            self.n_classes,
        )
        dynamic_data = (self.alpha_, self.W_sr_, self.L_, self.X_train_, self.y_train_)
        return (dynamic_data, static_data)

    @classmethod
    def tree_unflatten(cls, static_data, dynamic_data):
        alpha, W_sr, L, X_train, y_train = dynamic_data
        (kernel, max_iter_predict, poss, multi_class, n_classes) = static_data
        ins = cls(
            kernel=kernel,
            max_iter_predict=max_iter_predict,
            poss=poss,
            n_classes=n_classes,
            multi_class=multi_class,
        )
        ins.alpha_ = alpha
        ins.W_sr_ = W_sr
        ins.L_ = L
        ins.X_train_ = X_train
        ins.y_train_ = y_train
        return ins

    def fit(self, X, y):
        """Fit Gaussian process classification model.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Feature vectors of training data.

        y : jax numpy array (n_samples,) Target values,
        must be preprocessed to 0, 1, 2, ...

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if self.n_classes <= 2:
            estimator = BinaryEstimator(
                kernel=self.kernel,
                max_iter_predict=self.max_iter_predict,
                poss=self.poss,
            )
        elif self.multi_class == "one_vs_rest":
            estimator = OneVsRestEstimator(
                n_classes=self.n_classes,
                kernel=self.kernel,
                max_iter_predict=self.max_iter_predict,
                poss=self.poss,
            )
        elif self.multi_class == "one_vs_one":
            raise ValueError("one_vs_one classifier is not supported")
        else:
            raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        X = jnp.array(X)
        y = jnp.array(y)
        estimator.fit(X, y)

        state = estimator.state
        self.alpha_ = state[0]
        self.W_sr_ = state[1]
        self.L_ = state[2]

        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : jax numpy array (n_samples,)
            Predicted target values for X.
        """
        self.check_is_fitted()
        X = jnp.asarray(X)
        return predict(
            X,
            self.X_train_,
            self.alpha_,
            self.W_sr_,
            self.L_,
            self.n_classes,
            self.multi_class,
            self.kernel,
        )

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : jax numpy array (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order.
        """
        self.check_is_fitted()
        X = jnp.asarray(X)
        return predict_proba(
            X,
            self.X_train_,
            self.alpha_,
            self.W_sr_,
            self.L_,
            self.n_classes,
            self.multi_class,
            self.kernel,
        )

    def check_is_fitted(self):
        """Perform is_fitted validation for estimator."""
        assert self.alpha_ is not None, "Model should be fitted first."


jax.tree_util.register_pytree_node_class(GaussianProcessClassifier)
