# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jax
import jax.numpy as jnp

from sml.utils.extmath import newton_inv, standardize


@partial(jax.jit, static_argnames=("standardized"))
def vif(X: jax.Array, standardized: bool = False) -> jax.Array:
    """
    Compute Variance Inflation Factor (VIF) for multicollinearity detection.

    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity among predictor variables. Higher VIF values
    indicate stronger multicollinearity.

    VIF is calculated as 1/(1-R²) where R² is the coefficient of determination
    from regressing each predictor on all other predictors. This implementation
    uses the matrix inversion method: VIF = diagonal((X^T * X)^(-1)) * (n-1).

    Parameters
    ----------
    X : ArrayLike
        Input data matrix of shape (n_samples, n_features) where each column
        represents a predictor variable.
    standardized : bool, default=False
        Whether the input data is already standardized (mean=0, std=1).
        If False, the data will be standardized using z-score normalization.

    Returns
    -------
    vif_values : ArrayLike
        Array of VIF values with shape (n_features,). Each element represents
        the VIF for the corresponding predictor variable.

        Interpretation:
        - VIF = 1: No multicollinearity
        - 1 < VIF < 5: Moderate correlation, generally acceptable
        - VIF ≥ 5: High correlation, potential multicollinearity concern
        - VIF ≥ 10: Serious multicollinearity problem

    Notes
    -----
    VIF is commonly used in regression analysis to identify redundant
    predictor variables. Variables with high VIF values may need to be
    removed or combined to improve model stability.

    The calculation assumes that X has full column rank (no perfect
    multicollinearity). If the matrix is singular, the Newton inversion
    method will handle it appropriately.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sml.stats import vif
    >>> X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> vif_values = vif(X)
    >>> print(vif_values)
    """

    if not standardized:
        X = standardize(X)

    rows = X.shape[0]
    xTx = X.T @ X
    x_inv = newton_inv(xTx)
    res = jnp.diagonal(x_inv) * (rows - 1)
    return res
