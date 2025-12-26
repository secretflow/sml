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

    """

    def link(self, mu: jax.Array) -> jax.Array:
        return jnp.log(mu)

    def inverse(self, eta: jax.Array) -> jax.Array:
        return jnp.exp(eta)

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # d(log(mu))/dmu = 1/mu
        return 1.0 / mu

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # d(exp(eta))/deta = exp(eta)
        return jnp.exp(eta)


class LogitLink(Link):
    """
    The logit link function: g(mu) = log(mu / (1 - mu)).
    Canonical link for the Bernoulli/Binomial distribution.
    """

    def link(self, mu: jax.Array) -> jax.Array:
        return jnp.log(mu / (1.0 - mu))

    def inverse(self, eta: jax.Array) -> jax.Array:
        # Sigmoid function: 1 / (1 + exp(-eta))
        return 1.0 / (1.0 + jnp.exp(-eta))

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # d(log(mu/(1-mu)))/dmu = 1 / (mu * (1 - mu))
        return 1.0 / (mu * (1.0 - mu))

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # d(sigmoid(eta))/deta = sigmoid(eta) * (1 - sigmoid(eta))
        # Or equivalently: exp(eta) / (1 + exp(eta))^2
        exp_eta = jnp.exp(eta)
        return exp_eta / (1.0 + exp_eta) ** 2


class ProbitLink(Link):
    """
    The probit link function: g(mu) = Phi^(-1)(mu).
    Inverse of the standard normal CDF.

    Canonical link for approximated binary responses when using normal latent variables.
    """

    def link(self, mu: jax.Array) -> jax.Array:
        # Probit link: inverse of standard normal CDF
        # Phi^(-1)(p) = sqrt(2) * erf^(-1)(2p - 1)
        return jnp.sqrt(2.0) * jax.lax.erf_inv(2.0 * mu - 1.0)

    def inverse(self, eta: jax.Array) -> jax.Array:
        # Standard normal CDF: Phi(eta) = 0.5 * (1 + erf(eta/sqrt(2)))
        return 0.5 * (1.0 + jax.lax.erf(eta / jnp.sqrt(2.0)))

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # Derivative of probit link: 1 / phi(g(mu))
        # where phi is the standard normal PDF
        eta = self.link(mu)
        # Standard normal PDF: phi(eta) = exp(-eta^2/2) / sqrt(2*pi)
        phi_eta = jnp.exp(-0.5 * eta**2) / jnp.sqrt(2.0 * jnp.pi)
        return 1.0 / phi_eta

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # Derivative of standard normal CDF
        phi_eta = jnp.exp(-0.5 * eta**2) / jnp.sqrt(2.0 * jnp.pi)
        return phi_eta


class CLogLogLink(Link):
    """
    The complementary log-log link function: g(mu) = log(-log(1 - mu)).

    Used for binary outcomes with asymmetric effects, particularly when
    the upper asymptote is approached more slowly than the lower one.
    """

    def link(self, mu: jax.Array) -> jax.Array:
        # CLogLog link: log(-log(1 - mu))
        return jnp.log(-jnp.log(1.0 - mu))

    def inverse(self, eta: jax.Array) -> jax.Array:
        # Inverse of CLogLog: 1 - exp(-exp(eta))
        return 1.0 - jnp.exp(-jnp.exp(eta))

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # Derivative of CLogLog: -1 / ((1 - mu) * log(1 - mu))
        return -1.0 / ((1.0 - mu) * jnp.log(1.0 - mu))

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # Derivative of inverse CLogLog: exp(eta - exp(eta))
        exp_eta = jnp.exp(eta)
        return exp_eta * jnp.exp(-exp_eta)


class PowerLink(Link):
    """
    The power link function: g(mu) = mu^power for power != 0.

    Special cases:
    - power = 1: Identity link
    - power = 0: Log link (limit as power -> 0)
    - power = -1: Reciprocal link
    - power = 2: Square link
    - power = -2: Reciprocal squared link (canonical for Inverse Gaussian)

    Parameters
    ----------
    power : float
        The power parameter. Must not be 0 for direct use (use LogLink for power=0).
    """

    def __init__(self, power: float = 1.0):
        self.power = power
        if abs(power) < 1e-10:
            raise ValueError(
                f"PowerLink power cannot be 0. Use LogLink instead. Got {power}"
            )

    def link(self, mu: jax.Array) -> jax.Array:
        # Power link: mu^power
        return jnp.power(mu, self.power)

    def inverse(self, eta: jax.Array) -> jax.Array:
        # Inverse power link: eta^(1/power)
        return jnp.power(eta, 1.0 / self.power)

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # Derivative of power link: power * mu^(power-1)
        return self.power * jnp.power(mu, self.power - 1.0)

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # Derivative of inverse: (1/power) * eta^(1/power - 1)
        return (1.0 / self.power) * jnp.power(eta, 1.0 / self.power - 1.0)


class ReciprocalLink(Link):
    """
    The reciprocal link function: g(mu) = 1/mu.
    Special case of PowerLink with power = -1.
    Canonical link for Gamma distribution.
    """

    def link(self, mu: jax.Array) -> jax.Array:
        # Reciprocal link: 1/mu
        return 1.0 / mu

    def inverse(self, eta: jax.Array) -> jax.Array:
        # Inverse reciprocal: 1/eta
        return 1.0 / eta

    def link_deriv(self, mu: jax.Array) -> jax.Array:
        # Derivative of reciprocal: -1/mu^2
        return -1.0 / (mu**2)

    def inverse_deriv(self, eta: jax.Array) -> jax.Array:
        # Derivative of inverse reciprocal: -1/eta^2
        return -1.0 / (eta**2)
