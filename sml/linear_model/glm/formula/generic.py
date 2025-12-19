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


class GenericFormula(Formula):
    """
    Generic implementation of GLM component calculation.

    Uses standard textbook formulas for IRLS:
    - W = 1 / (V(mu) * (g'(mu))^2)
    - z_resid = (y - mu) * g'(mu)
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
        # 1. Compute linear predictor eta
        eta = X @ beta
        if offset is not None:
            eta += offset

        # Numerical stability: Clip eta
        if clip_eta is not None:
            eta = jnp.clip(eta, clip_eta[0], clip_eta[1])

        # 2. Compute mean mu via inverse link
        mu = family.link.inverse(eta)

        # Numerical stability: Clip mu
        if clip_mu is not None:
            mu = jnp.clip(mu, clip_mu[0], clip_mu[1])

        # 3. Compute atomic math components
        v_mu = family.distribution.unit_variance(mu)
        g_prime = family.link.link_deriv(mu)

        # 4. Compute Working Weights W
        # Fisher Information W = 1 / (V(mu) * (g'(mu))^2)
        # We add a small eps to denominators to prevent division by zero
        eps = 1e-12
        w = 1.0 / (v_mu * (g_prime**2) + eps)

        # Apply sample weights
        if sample_weight is not None:
            w = w * sample_weight

        # 5. Compute Working Residuals z_resid
        # z_resid = (y - mu) * g'(mu)
        z_resid = (y - mu) * g_prime

        # 6. Compute Deviance for monitoring/line search
        deviance = family.distribution.deviance(y, mu, sample_weight)

        # 7. Collect extras (like log-likelihood)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, mu, eta, deviance, extras
