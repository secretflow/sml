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

from .link import IdentityLink, Link, LogitLink, LogLink, PowerLink, ReciprocalLink


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
        where phi is the dispersion parameter (scale).

        Note on a(phi):
        In the exponential family form f(y; theta, phi) = exp((y*theta - b(theta))/a(phi) + c),
        typically a(phi) = phi / sample_weight.
        This class provides V(mu) = b''(theta).
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
        # R-style initialization: (y + mean(y)) / 2
        return (y + jnp.mean(y)) / 2.0

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
        eps = 1e-10
        dev = 2.0 * (
            y * jnp.log(y / mu + eps)
            + (1.0 - y) * jnp.log((1.0 - y) / (1.0 - mu) + eps)
        )
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # LL = y * log(mu) + (1-y) * log(1-mu)
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
        # Deviance formula for Poisson: D = 2 * sum(y * log(y/mu) - (y - mu))
        eps = 1e-10
        dev = 2.0 * (y * jnp.log(y / mu + eps) - (y - mu))
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # LL = y * log(mu) - mu - log(y!)
        # We omit log(y!) here as it is constant w.r.t parameters.
        ll = y * jnp.log(mu) - mu
        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # R-style initialization: (y + mean(y)) / 2
        return (y + jnp.mean(y)) / 2.0

    def get_canonical_link(self) -> Link:
        return LogLink()


class Gamma(Distribution):
    """
    Gamma distribution (for continuous positive data).
    Canonical link: Inverse (often Log is used in practice).
    """

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = mu^2
        return mu**2

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance = 2 * sum( (y - mu)/mu - log(y/mu) )
        eps = 1e-10
        dev = 2.0 * ((y - mu) / mu - jnp.log(y / mu + eps))
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Simplified LL: -(y/mu + log(mu))
        ll = -1.0 * (y / mu + jnp.log(mu))
        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # R-style initialization: (y + mean(y)) / 2
        return (y + jnp.mean(y)) / 2.0

    def get_canonical_link(self) -> Link:
        return ReciprocalLink()


class Tweedie(Distribution):
    """
    Tweedie distribution.
    A compound Poisson-Gamma distribution for modeling non-negative data.

    The Tweedie distribution includes:
    - Normal (p=0)
    - Poisson (p=1)
    - Gamma (p=2)
    - InverseGaussian (p=3)

    For 1 < p < 2, it's a compound Poisson-Gamma distribution.

    Parameters
    ----------
    power : float
        The variance power parameter p.
        Must be: p <= 0, p = 1, p >= 2, or 1 < p < 2
    """

    def __init__(self, power: float = 1.5):
        self.power = power
        # Validate power parameter
        if not (power <= 0 or power == 1 or power >= 2 or (1 < power < 2)):
            raise ValueError(
                f"Tweedie power must be <= 0, = 1, >= 2, or between 1 and 2. Got {power}"
            )

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = mu^p
        return jnp.power(mu, self.power)

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance for Tweedie depends on power parameter
        eps = 1e-10
        p = self.power

        if abs(p - 0) < 1e-10:  # Normal
            dev = (y - mu) ** 2
        elif abs(p - 1) < 1e-10:  # Poisson
            dev = 2.0 * (y * jnp.log(y / mu + eps) - (y - mu))
        elif abs(p - 2) < 1e-10:  # Gamma
            dev = 2.0 * ((y - mu) / mu - jnp.log(y / mu + eps))
        elif abs(p - 3) < 1e-10:  # Inverse Gaussian
            dev = (y - mu) ** 2 / (mu**2 * y + eps)
        elif 1 < p < 2:  # Compound Poisson-Gamma
            # Use approximation for compound distribution
            # This is generalized deviance
            dev = 2.0 * (
                y ** (2 - p) / ((2 - p) * mu ** (1 - p) + eps)
                - y ** (2 - p) / ((1 - p) * mu ** (2 - p) + eps)
                + mu ** (2 - p) / ((2 - p) * (1 - p))
            )
        else:
            # General case using approximation
            dev = jnp.abs(y - mu) ** (2 - p) / (mu ** (1 - p) + eps)

        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Log-likelihood approximation for Tweedie
        p = self.power

        if abs(p - 0) < 1e-10:  # Normal
            ll = -0.5 * (y - mu) ** 2 - 0.5 * jnp.log(2 * jnp.pi)
        elif abs(p - 1) < 1e-10:  # Poisson
            ll = y * jnp.log(mu) - mu
        elif abs(p - 2) < 1e-10:  # Gamma
            ll = -y / mu - jnp.log(mu)
        else:
            # Approximation for other powers
            ll = -(y ** (2 - p)) / ((2 - p) * mu ** (1 - p) + 1e-10)

        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # R-style initialization
        return (y + jnp.mean(y)) / 2.0

    def get_canonical_link(self) -> Link:
        # Canonical link depends on power
        p = self.power
        if abs(p - 0) < 1e-10:
            return IdentityLink()
        elif abs(p - 1) < 1e-10:
            return LogLink()
        elif abs(p - 2) < 1e-10:
            # For Gamma, canonical link is inverse (1/mu)
            # But inverse can be numerically unstable
            # We'll implement it as ReciprocalLink
            return ReciprocalLink()
        elif abs(p - 3) < 1e-10:
            # For Inverse Gaussian, canonical link is 1/mu^2
            # We'll use PowerLink(power=-2)
            return PowerLink(power=-2.0)
        else:
            # For other powers, use LogLink as a stable default
            return LogLink()


class NegativeBinomial(Distribution):
    """
    Negative Binomial distribution.
    For count data with overdispersion relative to Poisson.

    Parameters
    ----------
    alpha : float
        The dispersion parameter (also called size or nb_k).
        Larger alpha indicates more overdispersion.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        if alpha <= 0:
            raise ValueError(f"NegativeBinomial alpha must be positive. Got {alpha}")

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = mu + alpha * mu^2
        return mu + self.alpha * mu**2

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance for Negative Binomial
        eps = 1e-10
        alpha = self.alpha

        # Avoid division by zero
        mu_alpha = mu * alpha
        y_alpha = y * alpha

        dev = 2.0 * (
            y * jnp.log(y / mu + eps)
            - (y + 1.0 / alpha) * jnp.log((y_alpha + 1.0) / (mu_alpha + 1.0) + eps)
        )

        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Log-likelihood for Negative Binomial
        eps = 1e-10
        alpha = self.alpha
        alpha_mu = alpha * mu

        # Simplified log-likelihood (omitting constant terms)
        ll = y * jnp.log(alpha_mu / (1.0 + alpha_mu) + eps) - (1.0 / alpha) * jnp.log(
            1.0 + alpha_mu
        )

        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # R-style initialization
        return (y + jnp.mean(y)) / 2.0

    def get_canonical_link(self) -> Link:
        # Canonical link for Negative Binomial is log
        return LogLink()


class InverseGaussian(Distribution):
    """
    Inverse Gaussian (Wald) distribution.
    For positive continuous data with long tails.
    Canonical link: 1/mu^2
    """

    def unit_variance(self, mu: jax.Array) -> jax.Array:
        # V(mu) = mu^3
        return mu**3

    def deviance(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Deviance for Inverse Gaussian
        eps = 1e-10
        dev = (y - mu) ** 2 / (mu**2 * y + eps)
        return jnp.sum(dev * weights) if weights is not None else jnp.sum(dev)

    def log_likelihood(
        self, y: jax.Array, mu: jax.Array, weights: jax.Array | None = None
    ) -> jax.Array:
        # Log-likelihood for Inverse Gaussian (simplified)
        ll = -0.5 * (
            (y - mu) ** 2 / (mu**2 * y + 1e-10) + 1.5 * jnp.log(y) + jnp.log(mu)
        )
        return jnp.sum(ll * weights) if weights is not None else jnp.sum(ll)

    def starting_mu(self, y: jax.Array) -> jax.Array:
        # R-style initialization
        return (y + jnp.mean(y)) / 2.0

    def get_canonical_link(self) -> Link:
        # Canonical link is 1/mu^2
        return PowerLink(power=-2.0)
