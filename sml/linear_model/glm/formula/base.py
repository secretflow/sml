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

from typing import Any, Dict, Optional, Protocol, Tuple

import jax
from sml.linear_model.glm.core.family import Family


class Formula(Protocol):
    """
    Protocol for GLM component calculation strategies.

    A Formula defines how to compute the working weights and working residuals
    needed by the IRLS/Fisher Scoring solvers and SGD solvers.

    Mathematical Context:
    ---------------------
    The log-likelihood gradient (Score) w.r.t beta is:
        dL/dbeta = X.T * (y - mu) / (V(mu) * g'(mu))

    To unify IRLS and SGD, we decompose this term into two components:
    1. Working Weights (W):
       W = 1 / (V(mu) * g'(mu)^2)
       This is the diagonal of the Fisher Information Matrix (I = X.T * W * X).

    2. Working Residuals (z_resid):
       z_resid = (y - mu) * g'(mu)

    With these definitions:
    - Gradient (Score) = X.T * (W * z_resid)
    - Fisher Info (Hessian) = X.T * W * X
    - Adjusted Response (z) = eta + z_resid
    """

    def compute_components(
        self,
        X: jax.Array,
        y: jax.Array,
        beta: jax.Array,
        offset: Optional[jax.Array],
        family: Family,
        sample_weight: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, Dict[str, Any]]:
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