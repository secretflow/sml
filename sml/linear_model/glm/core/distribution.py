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

from .link import IdentityLink, Link, LogitLink, LogLink


class Distribution(ABC):
    """
    Abstract base class for GLM distributions.

    Provides methods to compute the unit variance function V(mu),
    deviance, log-likelihood, and starting values for the mean.
    """

    @abstractmethod
    def unit_variance(self, mu: jax.Array) -> jax.Array:
        """
        Compute the unit variance function V(mu).

        The variance of the distribution is given by: Var(Y) = phi * V(mu),
        where phi is the dispersion parameter.

        Parameters
        ----------
        mu : jax.Array
            The mean of the distribution.

        Returns
        -------
        v : jax.Array
            The computed variance function values.
        """
        pass

    @abstractmethod
    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        """
        Compute the deviance of the distribution.

        Deviance is a measure of goodness of fit, defined as:
        D = 2 * (LogLikelihood(Saturated Model) - LogLikelihood(Proposed Model))

        Parameters
        ----------
        y : jax.Array
            The target values.
        mu : jax.Array
            The predicted mean values.
        weights : jax.Array, optional
            Sample weights. If None, defaults to 1.

        Returns
        -------
        deviance : jax.Array
            The computed total deviance (scalar).
        """
        pass

    @abstractmethod
    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        """
        Compute the log-likelihood of the distribution.

        Used for calculating information criteria like AIC and BIC.

        Parameters
        ----------
        y : jax.Array
            The target values.
        mu : jax.Array
            The predicted mean values.
        weights : jax.Array, optional
            Sample weights.

        Returns
        -------
        ll : jax.Array
            The computed total log-likelihood (scalar).
        """
        pass

    @abstractmethod
    def starting_mu(self, y: jax.Array) -> jax.Array:
        """
        Compute robust starting values for the mean mu.

        These are used to initialize the IRLS or Newton solver.

        Parameters
        ----------
        y : jax.Array
            The target values.

        Returns
        -------
        mu_start : jax.Array
            Starting values for mu.
        """
        pass

    @abstractmethod
    def get_canonical_link(self) -> Link:
        """
        Return the default canonical link function for this distribution.

        Returns
        -------
        link : Link
            The canonical link instance.
        """
        pass


class Normal(Distribution):
    """
    Normal (Gaussian) distribution.
    Canonical link: Identity.
    """

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = 1
        return jnp.ones_like(mu)

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance = sum(weights * (y - mu)^2)
        dev = (y - mu) ** 2
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # LogLikelihood ~ -0.5 * sum((y - mu)^2) (ignoring constants and dispersion for basic LL)
        # Note: For exact AIC/BIC comparison, constant terms like log(2*pi) are included.
        ll = -0.5 * (y - mu) ** 2 - 0.5 * jnp.log(2 * jnp.pi)
        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # Start with the global mean
        return jnp.full_like(y, jnp.mean(y))

    def get_canonical_link(self) -> Link:
        return IdentityLink()


class Bernoulli(Distribution):
    """
    Bernoulli distribution (for binary classification).
    Canonical link: Logit.
    """

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = mu * (1 - mu)
        return mu * (1.0 - mu)

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance formula for Bernoulli
        # D = 2 * sum(y * log(y/mu) + (1-y) * log((1-y)/(1-mu)))
        # Added small epsilon to denominators to prevent log(0) and division by zero.
        mu = jnp.clip(mu, 1e-7, 1 - 1e-7)
        eps = 1e-10
        # Use a numerically stable formulation:
        # When y=1, term is 2*log(1/mu). When y=0, term is 2*log(1/(1-mu)).
        # Here we follow the standard formulation with eps.
        dev = 2.0 * (
            y * jnp.log(y / mu + eps) + (1.0 - y) * jnp.log((1.0 - y) / (1.0 - mu) + eps)
        )
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # LL = y * log(mu) + (1-y) * log(1-mu)
        mu = jnp.clip(mu, 1e-7, 1 - 1e-7)
        ll = y * jnp.log(mu) + (1.0 - y) * jnp.log(1.0 - mu)
        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # Start with (y + 0.5) / 2 to avoid boundary issues [0, 1]
        return (y + 0.5) / 2.0

    def get_canonical_link(self) -> Link:
        return LogitLink()


class Poisson(Distribution):
    """
    Poisson distribution (for count data).
    Canonical link: Log.
    """

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = mu
        return mu

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance formula for Poisson
        # D = 2 * sum(y * log(y/mu) - (y - mu))
        # Clip mu to ensure positiveness
        mu = jnp.clip(mu, 1e-7, 1e14)
        eps = 1e-10
        dev = 2.0 * (y * jnp.log(y / mu + eps) - (y - mu))
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # LL = y * log(mu) - mu - log(y!)
        # We omit log(y!) here as it is constant w.r.t parameters.
        mu = jnp.clip(mu, 1e-7, 1e14)
        ll = y * jnp.log(mu) - mu
        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # Start with weighted mean of y
        return (y + jnp.mean(y)) / 2.0

    def get_canonical_link(self) -> Link:
        return LogLink()
