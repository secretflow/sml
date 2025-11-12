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


import jax
import jax.numpy as jnp
from jax.lax.linalg import cholesky
from jax.scipy.linalg import cho_solve
from jax.scipy.special import expit

from sml.gaussian_process.kernels import RBF


class _BinaryGaussianProcessClassifierLaplace:
    def __init__(
        self,
        kernel=None,
        *,
        poss="sigmoid",
        max_iter_predict=100,
        kernel_internal=None,
    ):
        self.kernel = kernel
        self.kernel_ = kernel_internal
        self._check_kernel()

        self.max_iter_predict = max_iter_predict
        self.poss = poss

        if self.poss == "sigmoid":
            self.approx_func = expit
        else:
            raise ValueError(
                f"Unsupported prior-likelihood function {self.poss}."
                "Please try the default dunction sigmoid."
            )

    @property
    def alpha(self) -> jax.Array:
        return self.y_train - self.pi_

    def tree_flatten(self):
        static_data = (self.kernel, self.kernel_, self.max_iter_predict, self.poss)
        dynamic_data = (self.X_train_, self.y_train, self.pi_, self.W_sr_, self.L_)
        return (dynamic_data, static_data)

    @classmethod
    def tree_unflatten(cls, static_data, dynamic_data):
        (
            X_train_,
            y_train,
            pi_,
            W_sr_,
            L_,
        ) = dynamic_data

        (kernel, kernel_, max_iter_predict, poss) = static_data

        ins = cls(
            kernel,
            poss=poss,
            max_iter_predict=max_iter_predict,
            kernel_internal=kernel_,
        )
        ins.pi_ = pi_
        ins.W_sr_ = W_sr_
        ins.L_ = L_
        ins.X_train_ = X_train_
        ins.y_train = y_train
        return ins

    def fit(self, X, y):
        self.X_train_ = jnp.asarray(X)

        self.y_train = y

        K = self.kernel_(self.X_train_)

        (
            self.pi_,
            self.W_sr_,
            self.L_,
        ) = self._posterior_mode(K)
        return self

    def _posterior_mode(self, K: jax.Array):
        # Based on Algorithm 3.1 of GPML
        f = jnp.zeros_like(
            self.y_train, dtype=jnp.float32
        )  # a warning is triggered if float64 is used

        for _ in range(self.max_iter_predict):
            # W = self.log_and_2grads_and_negtive(f, self.y_train)
            pi = self.approx_func(f)
            W = pi * (1 - pi)
            W_sqr = jnp.sqrt(W)
            W_sqr_K = W_sqr[:, jnp.newaxis] * K

            B = jnp.eye(W.shape[0]) + W_sqr_K * W_sqr
            L = cholesky(B, symmetrize_input=False)
            # b = W * f + self.log_and_grad(f, self.y_train)
            b = W * f + (self.y_train - pi)
            a = b - jnp.dot(
                W_sqr[:, jnp.newaxis] * cho_solve((L, True), jnp.eye(W.shape[0])),
                W_sqr_K.dot(b),
            )
            f = K.dot(a)

            # no early stop here...

        # for warm-start
        # self.f_cached = f
        return pi, W_sqr, L

    def _check_kernel(self):
        if self.kernel_ is not None:
            return
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = RBF()
        else:
            raise NotImplementedError("Only RBF kernel is supported now.")


jax.tree_util.register_pytree_node_class(_BinaryGaussianProcessClassifierLaplace)
