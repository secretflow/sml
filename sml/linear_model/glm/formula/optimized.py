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
        clip_eta: tuple[float, float] | None = None,
        clip_mu: tuple[float, float] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        eta = X @ beta
        if offset is not None:
            eta += offset

        if clip_eta is not None:
            eta = jnp.clip(eta, clip_eta[0], clip_eta[1])

        # Optimized: mu calculation
        mu = jnp.exp(eta)

        if clip_mu is not None:
            mu = jnp.clip(mu, clip_mu[0], clip_mu[1])

        # Optimized components for Poisson + Log
        # W = mu, z_resid = y/mu - 1
        # Note: Implicitly assumes phi=1 (standard for Poisson) and a(phi) = 1/weight.
        w = mu
        if sample_weight is not None:
            w = w * sample_weight

        # Using (y - mu) / mu for better precision or (y / mu - 1)
        # Note: Generic formula uses (y - mu) * (1/mu)
        z_resid = (y - mu) / (mu + 1e-12)

        deviance = family.distribution.deviance(y, mu, sample_weight)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras
