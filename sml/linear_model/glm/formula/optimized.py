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

from typing import Any

import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.family import Family

from .base import Formula


class NormalIdentityFormula(Formula):
    """
    Optimized implementation for Normal (Gaussian) distribution with Identity link.

    - Canonical link: Identity (g(mu) = mu)
    - mu = eta
    - V(mu) = 1
    - g'(mu) = 1
    - W = 1 / (1 * 1^2) = 1
    - z_resid = (y - mu) * 1 = y - mu

    This is standard linear regression via WLS (OLS with constant weights).
    """

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Identity link: mu = eta
        mu = eta

        # For Normal + Identity:
        # W = 1 (constant weights)
        # z_resid = y - mu
        n_samples = X.shape[0]
        w = jnp.ones(n_samples, dtype=X.dtype)
        if sample_weight is not None:
            w = w * sample_weight

        z_resid = y - mu

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras


class BernoulliLogitFormula(Formula):
    """
    Optimized implementation for Bernoulli distribution with Logit link.

    - Canonical link: Logit (g(mu) = log(mu / (1-mu)))
    - mu = 1 / (1 + exp(-eta)) = sigmoid(eta)
    - V(mu) = mu * (1 - mu)
    - g'(mu) = 1 / (mu * (1 - mu))
    - W = 1 / (V(mu) * g'(mu)^2) = mu * (1 - mu)
    - z_resid = (y - mu) * g'(mu) = (y - mu) / (mu * (1 - mu))

    This is logistic regression.
    """

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Logit link inverse: mu = sigmoid(eta)
        mu = jax.nn.sigmoid(eta)

        # For Bernoulli + Logit (canonical):
        # W = mu * (1 - mu)
        # z_resid = (y - mu) / (mu * (1 - mu))
        eps = 1e-12
        variance = mu * (1.0 - mu) + eps

        w = variance
        if sample_weight is not None:
            w = w * sample_weight

        z_resid = (y - mu) / variance

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras


class PoissonLogFormula(Formula):
    """
    Optimized implementation for Poisson distribution with Log link.

    - Canonical link: Log
    - mu = exp(eta)
    - V(mu) = mu
    - g'(mu) = 1/mu
    - W = 1 / (mu * (1/mu)^2) = mu
    - z_resid = (y - mu) * (1/mu) = y/mu - 1
    """

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Optimized: mu calculation
        mu = jnp.exp(eta)

        # Optimized components for Poisson + Log
        # W = mu, z_resid = y/mu - 1
        eps = 1e-12
        w = mu
        if sample_weight is not None:
            w = w * sample_weight

        z_resid = (y - mu) / (mu + eps)

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras


class GammaReciprocalFormula(Formula):
    """
    Optimized implementation for Gamma distribution with Reciprocal (canonical) link.

    - Canonical link: Reciprocal (g(mu) = 1/mu)
    - eta = 1/mu => mu = 1/eta
    - V(mu) = mu^2
    - g'(mu) = -1/mu^2
    - W = 1 / (mu^2 * (1/mu^2)^2) = mu^2
    - z_resid = (y - mu) * (-1/mu^2)

    Note: Reciprocal link can be numerically tricky. Prefer Log link for business use.
    """

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Reciprocal link inverse: mu = 1/eta
        eps = 1e-12
        mu = 1.0 / (eta + eps)

        # For Gamma + Reciprocal (canonical):
        # V(mu) = mu^2, g'(mu) = -1/mu^2
        # W = mu^2 (magnitude; sign handled by z_resid)
        w = mu**2
        if sample_weight is not None:
            w = w * sample_weight

        # z_resid = (y - mu) * (-1/mu^2)
        z_resid = -(y - mu) / (mu**2 + eps)

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras


class GammaLogFormula(Formula):
    """
    Optimized implementation for Gamma distribution with Log link.

    - Non-canonical link: Log (g(mu) = log(mu))
    - mu = exp(eta)
    - V(mu) = mu^2
    - g'(mu) = 1/mu
    - W = 1 / (mu^2 * (1/mu)^2) = 1
    - z_resid = (y - mu) * (1/mu) = y/mu - 1

    This combination is numerically stable and commonly used in practice.
    """

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Log link inverse: mu = exp(eta)
        mu = jnp.exp(eta)

        # For Gamma + Log:
        # W = 1 (constant!)
        # z_resid = (y - mu) / mu = y/mu - 1
        n_samples = X.shape[0]
        w = jnp.ones(n_samples, dtype=X.dtype)
        if sample_weight is not None:
            w = w * sample_weight

        eps = 1e-12
        z_resid = (y - mu) / (mu + eps)

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras


class TweedieLogFormula(Formula):
    """
    Optimized implementation for Tweedie distribution with Log link.

    - Log link (common default for Tweedie)
    - mu = exp(eta)
    - V(mu) = mu^p (where p is the power parameter)
    - g'(mu) = 1/mu
    - W = 1 / (mu^p * (1/mu)^2) = mu^(2-p)
    - z_resid = (y - mu) / mu

    Note: This formula requires access to the power parameter from the distribution.
    """

    def __init__(self, power: float = 1.5):
        """
        Initialize TweedieLogFormula with power parameter.

        Parameters
        ----------
        power : float
            The Tweedie power parameter p.
            - p=0: Normal
            - p=1: Poisson
            - 1<p<2: Compound Poisson-Gamma
            - p=2: Gamma
            - p=3: Inverse Gaussian
        """
        self.power = power

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Log link inverse: mu = exp(eta)
        mu = jnp.exp(eta)

        # For Tweedie + Log:
        # W = mu^(2-p)
        # z_resid = (y - mu) / mu
        p = self.power
        eps = 1e-12

        w = jnp.power(mu + eps, 2.0 - p)
        if sample_weight is not None:
            w = w * sample_weight

        z_resid = (y - mu) / (mu + eps)

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras
