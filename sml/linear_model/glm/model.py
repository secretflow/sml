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

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.distribution import Distribution, Bernoulli
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.core.link import Link
from sml.linear_model.glm.metrics import metrics as metric_funcs
from sml.linear_model.glm.solvers import (
    IRLSSolver,
    SGDSolver,
    split_coef,
    Solver,
    get_registered_solver,
)


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
        - 'sgd': Stochastic Gradient Descent.
    max_iter : int, default=10
        Maximum number of iterations for the solver (or epochs for SGD).
    tol : float, default=1e-3
        Convergence tolerance for the solver.
        If 0, then the early stopping based on tolerance is disabled.
    stopping_rule : str, default='beta'
        The stopping rule to use for convergence. Options:
        - 'beta': based on coefficient change.
    learning_rate : float, default=1e-1
        Learning rate for SGD solver.
    decay_rate : float, default=1.0
        Learning rate decay factor for SGD (1.0 means no decay).
    decay_steps : int, default=1
        Decay steps for SGD learning rate schedule, counted by epochs.
    batch_size : int, default=1024
        Batch size for SGD solver.
    l2 : float, default=0.0
        L2 regularization strength (Ridge penalty).
        Note: we assume the dispersion of the distribution is always 1, and the functionality of dispersion will be added by l2 regularization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(
        self,
        dist: Distribution,
        link: Link | None = None,
        solver: str = "irls",
        max_iter: int = 10,
        tol: float = 1e-3,
        stopping_rule: str = "beta",
        learning_rate: float = 1e-1,
        decay_rate: float = 1.0,
        decay_steps: int = 1,
        batch_size: int = 1024,
        l2: float = 0.0,
        fit_intercept: bool = True,
    ):
        self.dist = dist
        self.link = link
        self.solver_name = solver
        self.max_iter = max_iter
        self.tol = tol
        self.stopping_rule = stopping_rule
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.l2 = l2
        self.fit_intercept = fit_intercept

        self.family_: Family = Family(self.dist, self.link)

        # Fitted attributes
        self.coef_: jax.Array | None = None
        self.intercept_: jax.Array | None = None

        # other attributes
        self._enable_spu_cache = False
        self._enable_spu_reveal = False
        self.history_: dict[str, Any] | None = None
        self.dispersion_: jax.Array | None = None
        self.scale_: float | jax.Array = 1.0

        # todo: add more parameter checks
        assert max_iter >= 1, "max_iter must be at least 1"

    def _get_solver(self) -> Solver:
        if self.solver_name == "irls":
            solver = get_registered_solver(self.family_)
            if solver is not None:
                return solver

            # fall back to default IRLS if no registered solver
            return IRLSSolver()
        elif self.solver_name == "sgd":
            # no registered solver for sgd yet
            return SGDSolver()
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")

    def fit(
        self,
        X: jax.Array,
        y: jax.Array,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
        y_scale: float | None = None,
        enable_spu_cache: bool = False,
        enable_spu_reveal: bool = False,
    ) -> "GLM":
        """
        Fit the GLM model.

        Note on MPC Stability and Scaling:
        ----------------------------------
        In MPC (Secure Multi-Party Computation) environments, such as SecretFlow SPU,
        fixed-point arithmetic can lead to overflows if 'y' has a large range.
        It is HIGHLY RECOMMENDED to normalize 'y' manually (e.g., y_scaled = y / y_scale)
        so that values are within a small range (e.g., [0, 1] or mean approx 1).

        Impact of Scaling (y_new = y / y_scale) on Coefficients:
        1. Log Link (Tweedie, Gamma, Poisson):
           - Slopes (beta_i, i>0): INVARIANT. They remain exactly the same as training on raw y.
           - Intercept (beta_0): Shifts by -log(y_scale).
           - Regularization: Usually no need to adjust `l2`.
        2. Identity Link (Gaussian):
           - All Coefficients: Scaled by 1/y_scale.
           - Regularization: `l2` should be scaled down approx by 1/y_scale^2 to avoid underfitting.
        3. Logit Link:
           - NOT RECOMMENDED. Keep y in {0, 1}.

        Parameters
        ----------
        X : jax.Array
            Training data of shape (n_samples, n_features).
        y : jax.Array
            Target values of shape (n_samples,).
        offset : jax.Array, optional
            A fixed offset added to the linear predictor.
            Shape must be broadcastable to (n_samples,).
        sample_weight : jax.Array, optional
            Individual weights for each sample.
        y_scale : float, optional, if None defaults to jnp.max(y)/2
            Scaling factor for 'y' (target variable).
            The model will be trained on `y / y_scale`.
            Predictions will be automatically rescaled by `mu * y_scale`.
        enable_spu_cache : bool, default=False
            Whether to enable SPU cached variables for intermediate computations.
        enable_spu_reveal : bool, default=False
            Whether to reveal intermediate results in SPU for higher performance.
            EXPERIMENTAL ONLY: will leak some information.

        Returns
        -------
        self : GLM
            Fitted estimator.
        """
        assert X.ndim == 2, "X must be a 2D array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert y.ndim in (1, 2), "y must be a 1D or 2D array"
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must be a column vector if 2D"
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        if offset is not None:
            assert (
                offset.shape[0] == X.shape[0]
            ), "offset must have the same number of samples as X"
            assert offset.ndim in (1, 2), "offset must be 1D or 2D array"
            if offset.ndim == 2:
                assert offset.shape[1] == 1, "offset must be a column vector if 2D"
            if offset.ndim == 1:
                offset = offset.reshape((-1, 1))
        if sample_weight is not None:
            assert (
                sample_weight.shape[0] == X.shape[0]
            ), "sample_weight must have the same number of samples as X"
            assert sample_weight.ndim in (1, 2), "sample_weight must be 1D or 2D array"
            if sample_weight.ndim == 2:
                assert (
                    sample_weight.shape[1] == 1
                ), "sample_weight must be a column vector if 2D"
            if sample_weight.ndim == 1:
                sample_weight = sample_weight.reshape((-1, 1))

        self._enable_spu_cache = enable_spu_cache
        self._enable_spu_reveal = enable_spu_reveal
        solver_impl = self._get_solver()

        # We train on y / y_scale
        y_scaled = None
        if isinstance(self.dist, Bernoulli):
            # For Bernoulli, scaling y does not make sense
            self.scale_ = 1.0
            y_scaled = y
        else:
            if y_scale is None:
                scale = jnp.max(y) / 2
            else:
                scale = y_scale
            self.scale_ = scale
            y_scaled = y / scale

        if offset is not None:
            offset = offset.reshape((-1, 1))

        if sample_weight is not None:
            sample_weight = sample_weight.reshape((-1, 1))

        # 5. Solve
        beta, dispersion, history = solver_impl.solve(
            X=X,
            y=y_scaled,
            family=self.family_,
            fit_intercept=self.fit_intercept,
            offset=offset,
            sample_weight=sample_weight,
            l2=self.l2,
            max_iter=self.max_iter,
            tol=self.tol,
            stopping_rule=self.stopping_rule,
            learning_rate=self.learning_rate,
            decay_rate=self.decay_rate,
            decay_steps=self.decay_steps,
            batch_size=self.batch_size,
            enable_spu_cache=self._enable_spu_cache,
            enable_spu_reveal=self._enable_spu_reveal,
        )

        # 6. Store results
        self.coef_, self.intercept_ = split_coef(beta, self.fit_intercept)
        self.dispersion_ = dispersion
        self.history_ = history

        return self

    def predict(self, X: jax.Array, offset: jax.Array | None = None) -> jax.Array:
        """
        Predict mean values.

        mu = link.inverse(X @ coef + intercept + offset) * scale

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

        # Inverse link to get scaled mu (since model was trained on scaled y)
        mu_scaled = self.family_.link.inverse(eta)

        # Rescale back to original scale
        return mu_scaled * self.scale_

    def predict_linear(
        self, X: jax.Array, offset: jax.Array | None = None
    ) -> jax.Array:
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

        if offset is not None:
            eta += offset

        return eta

    def score(
        self,
        X: jax.Array,
        y: jax.Array,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
    ) -> jax.Array:
        """
        Compute the deviance score (lower is better).

        Parameters
        ----------
        X : jax.Array
            Data to score.
        y : jax.Array
            True targets.
        offset : jax.Array, optional
            Offset.
        sample_weight : jax.Array, optional
            Weights.

        Returns
        -------
        score : jax.Array
            Negative Deviance.
        """
        if self.family_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Predict uses scale to return mu in original space
        mu = self.predict(X, offset=offset)

        # Calculate deviance in original space
        deviance = self.family_.distribution.deviance(y, mu, sample_weight)
        return -deviance

    def evaluate(
        self,
        X: jax.Array,
        y: jax.Array,
        metrics: Sequence[str] = ("deviance", "aic", "rmse"),
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """
        Evaluate the model using multiple metrics.

        Parameters
        ----------
        X : jax.Array
            Data to evaluate.
        y : jax.Array
            True targets.
        metrics : Sequence[str]
            List of metrics to compute. Options: 'deviance', 'aic', 'bic', 'rmse', 'auc'.
        offset : jax.Array, optional
            Offset.
        sample_weight : jax.Array, optional
            Weights.

        Returns
        -------
        results : Dict[str, jax.Array]
            Dictionary of computed metrics.
        """
        if self.family_ is None or self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Get predictions in original space
        mu = self.predict(X, offset=offset)
        n_samples = X.shape[0]
        # Rank = features + intercept
        rank = self.coef_.shape[0] + (1 if self.fit_intercept else 0)

        results = {}
        for metric in metrics:
            if metric == "deviance":
                results[metric] = metric_funcs.deviance(
                    y, mu, self.family_, sample_weight
                )
            elif metric == "aic":
                results[metric] = metric_funcs.aic(
                    y, mu, self.family_, rank, sample_weight
                )
            elif metric == "bic":
                results[metric] = metric_funcs.bic(
                    y, mu, self.family_, rank, n_samples, sample_weight
                )
            elif metric == "rmse":
                results[metric] = metric_funcs.rmse(y, mu, sample_weight)
            elif metric == "auc":
                results[metric] = metric_funcs.auc(y, mu, sample_weight)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return results
