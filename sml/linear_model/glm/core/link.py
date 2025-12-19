# Copyright 2024 Ant Group Co., Ltd.
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

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class Link(ABC):
    """
    Abstract base class for GLM link functions.

    The link function g(.) relates the mean of the response mu to the linear predictor eta:
    eta = g(mu)
    mu = g^{-1}(eta)
    """

    @abstractmethod
    def link(self, mu: jax.Array) -> jax.Array:
        """
        Compute the link function g(mu).

        Parameters
        ----------
        mu : jax.Array
            The mean of the response variable.

        Returns
        -------
        eta : jax.Array
            The linear predictor.
        """
        pass

    @abstractmethod
    def inverse(self, eta: jax.Array) -> jax.Array:
        """
        Compute the inverse link function g^{-1}(eta).

        Parameters
        ----------
        eta : jax.Array
            The linear predictor.

        Returns
        -------
        mu : jax.Array
            The mean of the response variable.
        """
        pass

    @abstractmethod
    def link_deriv(self, mu: jax.Array) -> jax.Array:
        """
        Compute the derivative of the link function d_eta / d_mu.

        Parameters
        ----------
        mu : jax.Array
            The mean of the response variable.

        Returns
        -------
        deriv : jax.Array
            The derivative of the link function with respect to mu.
        """
        pass

    @abstractmethod
    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        """
        Compute the derivative of the inverse link function d_mu / d_eta.

        Parameters
        ----------
        eta : jax.Array
            The linear predictor.

        Returns
        -------
        deriv : jax.Array
            The derivative of the inverse link function with respect to eta.
        """
        pass


class IdentityLink(Link):
    """
    The identity link function: g(mu) = mu.
    Canonical link for the Normal distribution.
    """

    def link(self, mu: jax.Array) -> jax.Array:
        return mu

    def inverse(self, eta: jax.Array) -> jax.Array:
        return eta

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        return jnp.ones_like(mu)

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        return jnp.ones_like(eta)


class LogLink(Link):
    """
    The log link function: g(mu) = log(mu).
    Canonical link for Poisson, Gamma, Tweedie (often) distributions.

    Parameters
    ----------
    clip_mu : Tuple[float, float], optional
        The lower and upper bounds to clip mu for numerical stability,
        avoiding log(0) or overflow. Default is (1e-7, 1e14).
    """

    def __init__(self, clip_mu: tuple[float, float] = (1e-7, 1e14)):
        self.clip_mu = clip_mu

    def link(self, mu: jax.Array) -> jax.Array:
        # Clip mu to avoid log(<=0) which results in nan or -inf
        mu = jnp.clip(mu, self.clip_mu[0], self.clip_mu[1])
        return jnp.log(mu)

    def inverse(self, eta: jax.Array) -> jax.Array:
        # We don't clip eta here by default; upper layers (like solver or formula)
        # can handle eta clipping if necessary to prevent exp() overflow.
        return jnp.exp(eta)

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # d(log(mu))/dmu = 1/mu
        mu = jnp.clip(mu, self.clip_mu[0], self.clip_mu[1])
        return 1.0 / mu

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # d(exp(eta))/deta = exp(eta)
        return jnp.exp(eta)


class LogitLink(Link):
    """
    The logit link function: g(mu) = log(mu / (1 - mu)).
    Canonical link for the Bernoulli/Binomial distribution.

    Parameters
    ----------
    clip_mu : Tuple[float, float], optional
        The lower and upper bounds to clip mu to avoid division by zero
        in the logit function. Default is (1e-7, 1 - 1e-7).
    """

    def __init__(self, clip_mu: tuple[float, float] = (1e-7, 1 - 1e-7)):
        self.clip_mu = clip_mu

    def link(self, mu: jax.Array) -> jax.Array:
        # Clip mu to avoid log(0) or division by zero
        mu = jnp.clip(mu, self.clip_mu[0], self.clip_mu[1])
        return jnp.log(mu / (1.0 - mu))

    def inverse(self, eta: jax.Array) -> jax.Array:
        # Sigmoid function: 1 / (1 + exp(-eta))
        # Using specific implementation for better numerical stability could be added,
        # but jax.nn.sigmoid is also an option. Here we use the raw formula.
        return 1.0 / (1.0 + jnp.exp(-eta))

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # d(log(mu/(1-mu)))/dmu = 1 / (mu * (1 - mu))
        mu = jnp.clip(mu, self.clip_mu[0], self.clip_mu[1])
        return 1.0 / (mu * (1.0 - mu))

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # d(sigmoid(eta))/deta = sigmoid(eta) * (1 - sigmoid(eta))
        # Or equivalently: exp(eta) / (1 + exp(eta))^2
        exp_eta = jnp.exp(eta)
        return exp_eta / (1.0 + exp_eta) ** 2
