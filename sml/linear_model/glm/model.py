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

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from sml.linear_model.glm.core.distribution import Distribution
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.core.link import Link
from sml.linear_model.glm.formula.base import Formula
from sml.linear_model.glm.formula.dispatch import dispatcher
from sml.linear_model.glm.metrics import metrics as metric_funcs
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.irls import IRLSSolver
from sml.linear_model.glm.solvers.sgd import SGDSolver
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
        - 'sgd': Stochastic Gradient Descent.
    max_iter : int, default=100
        Maximum number of iterations for the solver (or epochs for SGD).
    tol : float, default=1e-4
        Convergence tolerance for the solver.
    learning_rate : float, default=1e-2
        Learning rate for SGD solver.
    decay_rate : float, default=1.0
        Learning rate decay factor for SGD (1.0 means no decay).
    decay_steps : int, default=100
        Decay steps for SGD learning rate schedule.
    batch_size : int, default=128
        Batch size for SGD solver.
    l2 : float, default=0.0
        L2 regularization strength (Ridge penalty).
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    random_state : int, optional, default=None
        Seed for JAX PRNG (e.g., for SGD initialization or shuffling).
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
        link: Optional[Link] = None,
        solver: str = "irls",
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 1e-2,
        decay_rate: float = 1.0,
        decay_steps: int = 100,
        batch_size: int = 128,
        l2: float = 0.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        formula: Optional[Formula] = None,
        clip_eta: Optional[Tuple[float, float]] = None,
        clip_mu: Optional[Tuple[float, float]] = None,
    ):
        self.dist = dist
        self.link = link
        self.solver_name = solver
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.formula = formula
        self.clip_eta = clip_eta
        self.clip_mu = clip_mu

        # Fitted attributes
        self.family_: Optional[Family] = None
        self.coef_: Optional[jax.Array] = None
        self.intercept_: Optional[jax.Array] = None
        self.dispersion_: Optional[jax.Array] = None
        self.history_: Optional[Dict[str, Any]] = None

    def _get_solver(self) -> Solver:
        if self.solver_name == "irls":
            return IRLSSolver()
        elif self.solver_name == "sgd":
            return SGDSolver()
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")

    def fit(
        self,
        X: jax.Array,
        y: jax.Array,
        offset: Optional[jax.Array] = None,
        sample_weight: Optional[jax.Array] = None,
        scale: float = 1.0,
    ) -> "GLM":
        """
        Fit the GLM model.

        Note on MPC Stability:
        ----------------------
        In MPC (Secure Multi-Party Computation) environments, such as SecretFlow SPU,
        fixed-point arithmetic can lead to overflows if 'y' has a large range.
        It is HIGHLY RECOMMENDED to normalize 'y' manually (e.g., y_scaled = y / scale)
        so that values are within a small range (e.g., [0, 1] or mean approx 1).

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
        scale : float, default=1.0
            Scaling factor for 'y' (target variable).
            The model will be trained on `y / scale`.
            Predictions will be automatically rescaled by `mu * scale`.

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

        # 4. Apply Scaling
        # We train on y / scale
        y_scaled = y / scale
        
        # 5. Solve
        beta, dispersion, history = solver_impl.solve(
            X=X,
            y=y_scaled,
            family=self.family_,
            formula=formula_impl,
            fit_intercept=self.fit_intercept,
            offset=offset,
            sample_weight=sample_weight,
            l2=self.l2,
            max_iter=self.max_iter,
            tol=self.tol,
            learning_rate=self.learning_rate,
            decay_rate=self.decay_rate,
            decay_steps=self.decay_steps,
            batch_size=self.batch_size,
            random_state=self.random_state,
            clip_eta=self.clip_eta,
            clip_mu=self.clip_mu,
        )

        # 6. Store results
        self.coef_, self.intercept_ = split_coef(beta, self.fit_intercept)
        self.dispersion_ = dispersion
        self.history_ = history

        return self

    def predict(
        self, 
        X: jax.Array, 
        offset: Optional[jax.Array] = None,
        scale: float = 1.0
    ) -> jax.Array:
        """
        Predict mean values.

        mu = link.inverse(X @ coef + intercept + offset) * scale

        Parameters
        ----------
        X : jax.Array
            Samples to predict.
        offset : jax.Array, optional
            Offset to add to the linear predictor.
        scale : float, default=1.0
            Scaling factor to denormalize the prediction.
            Should match the scale used in `fit`.

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
        return mu_scaled * scale

    def predict_linear(self, X: jax.Array, offset: Optional[jax.Array] = None) -> jax.Array:
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
        offset: Optional[jax.Array] = None, 
        sample_weight: Optional[jax.Array] = None,
        scale: float = 1.0
    ) -> jax.Array:
        """
        Compute the deviance score (lower is better).

        Parameters
        ----------
        scale : float, default=1.0
            Scale used for prediction denormalization.

        Returns
        -------
        score : jax.Array
            Negative Deviance.
        """
        if self.family_ is None:
             raise RuntimeError("Model is not fitted yet.")
        
        # Predict uses scale to return mu in original space
        mu = self.predict(X, offset=offset, scale=scale)
        
        # Calculate deviance in original space
        deviance = self.family_.distribution.deviance(y, mu, sample_weight)
        return -deviance
    
    def evaluate(
        self, 
        X: jax.Array, 
        y: jax.Array, 
        metrics: Sequence[str] = ("deviance", "aic", "rmse"),
        offset: Optional[jax.Array] = None,
        sample_weight: Optional[jax.Array] = None,
        scale: float = 1.0
    ) -> Dict[str, jax.Array]:
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
        scale : float, default=1.0
            Scaling factor to denormalize the prediction.
            
        Returns
        -------
        results : Dict[str, jax.Array]
            Dictionary of computed metrics.
        """
        if self.family_ is None or self.coef_ is None:
             raise RuntimeError("Model is not fitted yet.")
        
        # Get predictions in original space
        mu = self.predict(X, offset=offset, scale=scale)
        n_samples = X.shape[0]
        # Rank = features + intercept
        rank = self.coef_.shape[0] + (1 if self.fit_intercept else 0)
        
        results = {}
        for metric in metrics:
            if metric == "deviance":
                results[metric] = metric_funcs.deviance(y, mu, self.family_, sample_weight)
            elif metric == "aic":
                results[metric] = metric_funcs.aic(y, mu, self.family_, rank, sample_weight)
            elif metric == "bic":
                results[metric] = metric_funcs.bic(y, mu, self.family_, rank, n_samples, sample_weight)
            elif metric == "rmse":
                results[metric] = metric_funcs.rmse(y, mu, sample_weight)
            elif metric == "auc":
                results[metric] = metric_funcs.auc(y, mu, sample_weight)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results
