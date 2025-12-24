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

from sml.linear_model.glm.core.family import Family

from .base import Formula


class GenericFormula(Formula):
    """
    Generic implementation of GLM component calculation.

    Uses standard textbook formulas for IRLS:
    - W = 1 / (V(mu) * (g'(mu))^2)
    - z_resid = (y - mu) * g'(mu)
    """

    def _compute_w_and_z_resid(
        self,
        y: jax.Array,
        mu: jax.Array,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        """
        Internal helper to compute W, z_resid, deviance, and extras from mu.

        This is the core computation shared by both compute_components and
        compute_components_from_mu.
        """
        # 1. Compute atomic math components
        v_mu = family.distribution.unit_variance(mu)
        g_prime = family.link.link_deriv(mu)
        # TODO: add cache here for g_prime if needed

        # 2. Compute Working Weights W
        # Fisher Information W = 1 / (V(mu) * (g'(mu))^2)
        w = 1.0 / (v_mu * (g_prime**2))

        # Apply sample weights
        if sample_weight is not None:
            w = w * sample_weight

        # 3. Compute Working Residuals z_resid
        # z_resid = (y - mu) * g'(mu)
        z_resid = (y - mu) * g_prime

        # 4. Compute Deviance for monitoring/line search
        deviance = family.distribution.deviance(y, mu, sample_weight)

        # 5. Collect extras (like log-likelihood)
        extras = {
            "log_likelihood": family.distribution.log_likelihood(y, mu, sample_weight)
        }

        return w, z_resid, deviance, extras

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: jax.Array | None,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        # 1. Compute linear predictor eta
        eta = X @ beta
        if offset is not None:
            eta += offset

        # 2. Compute mean mu via inverse link
        mu = family.link.inverse(eta)

        # 3. Compute W, z_resid, deviance, extras
        w, z_resid, deviance, extras = self._compute_w_and_z_resid(
            y, mu, family, sample_weight
        )

        return w, z_resid, mu, eta, deviance, extras

    def compute_components_from_mu(
        self,
        y: jax.Array,
        mu: jax.Array,
        family: Family,
        sample_weight: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        # 1. Compute eta from mu via link function
        eta = family.link.link(mu)

        # 2. Compute W, z_resid, deviance, extras
        w, z_resid, deviance, extras = self._compute_w_and_z_resid(
            y, mu, family, sample_weight
        )

        return w, z_resid, eta, deviance, extras
