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

from typing import Any, Protocol

import jax

from sml.linear_model.glm.core.family import Family


class Formula(Protocol):
    """
    Protocol for GLM component calculation strategies.

    A Formula defines how to compute the working weights and working residuals
    needed by the IRLS/Fisher Scoring solvers.
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
        """
        Compute the components for a single solver iteration.

        Parameters
        ----------
        X : jax.Array
            Feature matrix of shape (n_samples, n_features).
        y : jax.Array
            Target vector of shape (n_samples,).
        beta : jax.Array
            Current coefficient vector (including intercept if any).
        offset : jax.Array, optional
            Offset vector added to the linear predictor.
        family : Family
            The GLM family (distribution + link).
        sample_weight : jax.Array, optional
            External weights per sample.
        clip_eta : Tuple[float, float], optional
            Bounds for clipping the linear predictor eta.
        clip_mu : Tuple[float, float], optional
            Bounds for clipping the mean mu.

        Returns
        -------
        W : jax.Array
            Working weights (Fisher Info weights).
        z_resid : jax.Array
            Working residuals (y - mu) * g'(mu).
        mu : jax.Array
            The computed mean mu.
        eta : jax.Array
            The computed linear predictor eta.
        deviance : jax.Array
            Total deviance at current parameters.
        extras : Dict[str, Any]
            Additional information (e.g., log-likelihood).
        """
        ...
