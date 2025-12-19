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

from sml.linear_model.glm.core.distribution import Distribution
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.core.link import Link
from sml.linear_model.glm.formula.base import Formula
from sml.linear_model.glm.formula.dispatch import dispatcher
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.irls import IRLSSolver
from sml.linear_model.glm.solvers.utils import split_coef


class GLM:
    """
    Generalized Linear Model (GLM) estimator.

    A JAX-based implementation of GLM that supports various distributions and link functions.
    It uses a modular architecture allowing custom formulas and solvers.

    Parameters
    ----------
    dist : Distribution
        The distribution of the target variable (e.g., Normal, Poisson, Bernoulli).
    link : Link, optional
        The link function. If None, the canonical link of the distribution is used.
    solver : str, default='irls'
        The solver algorithm to use. Currently supports:
        - 'irls': Iteratively Reweighted Least Squares.
    max_iter : int, default=100
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Convergence tolerance for the solver.
    l2 : float, default=0.0
        L2 regularization strength (Ridge penalty).
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    offset : jax.Array, optional
        A fixed offset added to the linear predictor.
    formula : Formula, optional
        Custom formula implementation. If None, it is resolved via the dispatcher.
    clip_eta : Tuple[float, float], optional
        Bounds to clip the linear predictor eta for numerical stability.
    clip_mu : Tuple[float, float], optional
        Bounds to clip the mean prediction mu for numerical stability.
    """

    def __init__(
        self,
        dist: Distribution,
        link: Link | None = None,
        solver: str = "irls",
        max_iter: int = 100,
        tol: float = 1e-4,
        l2: float = 0.0,
        fit_intercept: bool = True,
        offset: jax.Array | None = None,
        formula: Formula | None = None,
        clip_eta: tuple[float, float] | None = None,
        clip_mu: tuple[float, float] | None = None,
    ):
        self.dist = dist
        self.link = link
        self.solver_name = solver
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.offset = offset
        self.formula = formula
        self.clip_eta = clip_eta
        self.clip_mu = clip_mu

        # Fitted attributes
        self.family_: Family | None = None
        self.coef_: jax.Array | None = None
        self.intercept_: jax.Array | None = None
        self.dispersion_: jax.Array | None = None
        self.history_: dict[str, Any] | None = None

    def _get_solver(self) -> Solver:
        if self.solver_name == "irls":
            return IRLSSolver()
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")

    def fit(
        self,
        X: jax.Array,
        y: jax.Array,
        sample_weight: jax.Array | None = None,
    ) -> "GLM":
        """
        Fit the GLM model.

        Parameters
        ----------
        X : jax.Array
            Training data of shape (n_samples, n_features).
        y : jax.Array
            Target values of shape (n_samples,).
        sample_weight : jax.Array, optional
            Individual weights for each sample.

        Returns
        -------
        self : GLM
            Fitted estimator.
        """
        # 1. Setup Family
        self.family_ = Family(self.dist, self.link)

        # 2. Resolve Formula
        if self.formula is None:
            formula_impl = dispatcher.resolve(self.family_.distribution, self.family_.link)
        else:
            formula_impl = self.formula

        # 3. Get Solver
        solver_impl = self._get_solver()

        # 4. Solve
        beta, dispersion, history = solver_impl.solve(
            X=X,
            y=y,
            family=self.family_,
            formula=formula_impl,
            fit_intercept=self.fit_intercept,
            offset=self.offset,
            sample_weight=sample_weight,
            l2=self.l2,
            max_iter=self.max_iter,
            tol=self.tol,
            clip_eta=self.clip_eta,
            clip_mu=self.clip_mu,
        )

        # 5. Store results
        self.coef_, self.intercept_ = split_coef(beta, self.fit_intercept)
        self.dispersion_ = dispersion
        self.history_ = history

        return self

    def predict(self, X: jax.Array, offset: jax.Array | None = None) -> jax.Array:
        """
        Predict mean values.

        mu = link.inverse(X @ coef + intercept + offset)

        Parameters
        ----------
        X : jax.Array
            Samples to predict.
        offset : jax.Array, optional
            Offset to add to the linear predictor.

        Returns
        -------
        mu : jax.Array
            Predicted mean values.
        """
        eta = self.predict_linear(X, offset)
        if self.family_ is None:
             raise RuntimeError("Model is not fitted yet.")
        return self.family_.link.inverse(eta)

    def predict_linear(self, X: jax.Array, offset: jax.Array | None = None) -> jax.Array:
        """
        Predict linear predictor values (eta).

        eta = X @ coef + intercept + offset

        Parameters
        ----------
        X : jax.Array
            Samples to predict.
        offset : jax.Array, optional
            Offset to add to the linear predictor.

        Returns
        -------
        eta : jax.Array
            Predicted linear values.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet.")

        eta = X @ self.coef_ + self.intercept_

        # Add training offset if provided and no new offset provided?
        # Standard behavior: Use new offset if provided, otherwise use nothing (offsets are per-sample).
        if offset is not None:
            eta += offset

        return eta

    def score(self, X: jax.Array, y: jax.Array, sample_weight: jax.Array | None = None) -> jax.Array:
        """
        Compute the deviance score (lower is better).

        Note: In scikit-learn, score() typically returns R^2 or accuracy (higher is better).
        For GLMs, we return negative deviance so that higher is better, or just deviance.
        Here we return the negative deviance (log-likelihood proxy).

        Returns
        -------
        score : jax.Array
            Negative Deviance.
        """
        if self.family_ is None:
             raise RuntimeError("Model is not fitted yet.")
        mu = self.predict(X)
        deviance = self.family_.distribution.deviance(y, mu, sample_weight)
        return -deviance
