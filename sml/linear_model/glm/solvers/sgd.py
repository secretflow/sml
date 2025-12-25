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
from sml.linear_model.glm.solvers.utils import check_convergence
from sml.utils import sml_drop_cached_var, sml_make_cached_var, sml_reveal

from .base import Solver
from .utils import add_intercept


def sgd_epoch_update(
    beta: jax.Array,
    epoch: int,
    xs: list,
    ys: list,
    offs: list,
    sws: list,
    family: Family,
    n_features: int,
    l2: float,
    fit_intercept: bool,
    learning_rate: float,
    decay_rate: float,
    decay_steps: int,
) -> jax.Array:
    """
    Perform one epoch of SGD updates over all batches.

    Parameters
    ----------
    beta : jax.Array
        Current coefficient estimates, shape (n_features, 1).
    epoch : int
        Current epoch number (for learning rate decay).
    xs : list
        List of feature batches.
    ys : list
        List of target batches.
    offs : list
        List of offset batches (or None).
    sws : list
        List of sample weight batches (or None).
    family : Family
        GLM family containing distribution and link function.
    n_features : int
        Number of features (including intercept if applicable).
    l2 : float
        L2 regularization strength.
    fit_intercept : bool
        Whether intercept is included.
    learning_rate : float
        Base learning rate.
    decay_rate : float
        Learning rate decay rate.
    decay_steps : int
        Learning rate decay steps.

    Returns
    -------
    beta : jax.Array
        Updated coefficient estimates.
    """
    cur_lr = learning_rate * decay_rate ** (epoch / decay_steps)

    for x_batch, y_batch, offset_batch, sample_weight_batch in zip(
        xs, ys, offs, sws, strict=True
    ):
        actual_batch_size = x_batch.shape[0]
        eta = jnp.matmul(x_batch, beta)
        if offset_batch is not None:
            eta = eta + offset_batch

        # Compute mu from eta
        mu = family.link.inverse(eta)

        # Compute Error = mu - y
        err = mu - y_batch

        # Compute gradient components: (mu - y) / (g'(mu) * V(mu))
        g_prime_inv = 1 / family.link.link_deriv(mu)
        unit_var = family.distribution.unit_variance(mu)
        grad_components = err * g_prime_inv / unit_var

        if sample_weight_batch is not None:
            grad_components = grad_components * sample_weight_batch

        grad = jnp.matmul(x_batch.T, grad_components)

        # Apply L2 regularization (not on intercept)
        if l2 > 0:
            grad = grad + l2 * beta
            if fit_intercept:
                grad = grad.at[n_features - 1].set(
                    grad[n_features - 1, 0] - l2 * beta[n_features - 1, 0]
                )

        # Gradient descent update
        beta = beta - cur_lr * grad / actual_batch_size

    return beta


class SGDSolver(Solver):
    """
    Stochastic Gradient Descent (SGD) solver for GLMs.

    Mathematical Derivation:
    ------------------------
    We maximize the Log-Likelihood L(beta) (Gradient Ascent).

    1. Gradient of Log-Likelihood (per sample):
       dL_i / dbeta = (y_i - mu_i) / a(phi) * (d_theta / d_mu) * (d_mu / d_eta) * x_i
                    = (y_i - mu_i) * (1 / V(mu_i)) * (1 / g'(mu_i)) * x_i

    2. Relation to Formula Components:
       Formula computes:
       - W_i = 1 / (V(mu_i) * g'(mu_i)^2)
       - z_resid_i = (y_i - mu_i) * g'(mu_i)

       Multiplying them:
       W_i * z_resid_i = (y_i - mu_i) / (V(mu_i) * g'(mu_i))

       Therefore:
       Gradient_i = x_i * (W_i * z_resid_i)

    3. Update Rule:
       beta_{t+1} = beta_t + learning_rate * Gradient_Batch
       (Plus L2 regularization term gradient adjustment)

    4. Learning Rate Decay:
       lr_t = learning_rate * decay_rate ^ (step / decay_steps)

    Intercept-Only Initialization (sklearn/H2O style):
    --------------------------------------------------
    Instead of starting with beta=0 or full R-style least squares:
    1. Set all feature coefficients to 0
    2. Set intercept to link(mean(y))
    This is simpler, more efficient for SGD, and commonly used in practice.

    Supports Mini-Batch SGD via sequential slicing.
    """

    def solve(
        self,
        X: jax.Array,
        y: jax.Array,
        family: Family,
        fit_intercept: bool = True,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
        l2: float = 0.0,
        max_iter: int = 100,  # Interpreted as Epochs
        tol: float = 1e-4,
        stopping_rule: str = "beta",
        learning_rate: float = 1e-2,
        decay_rate: float = 1.0,
        decay_steps: int = 1,
        batch_size: int = 128,
        enable_spu_cache: bool = False,
        enable_spu_reveal: bool = False,  # used in IRLS
    ) -> tuple[jax.Array, jax.Array | None, dict[str, Any] | None]:
        is_early_stop_enabled = tol > 0.0

        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape
        if enable_spu_cache:
            X_train = sml_make_cached_var(X_train)

        # Handle batch size
        batch_size = min(batch_size, n_samples)
        n_batches = n_samples // batch_size

        # Pre-split data into batches
        xs = jnp.array_split(X_train, n_batches, axis=0)
        ys = jnp.array_split(y, n_batches, axis=0)
        if offset is not None:
            offs = jnp.array_split(offset, n_batches, axis=0)
        else:
            offs = [None] * n_batches
        if sample_weight is not None:
            sws = jnp.array_split(sample_weight, n_batches, axis=0)
        else:
            sws = [None] * n_batches

        # 2. Intercept-Only Initialization (sklearn/H2O style)
        # Initialize all coefficients to 0, then set intercept to link(mean(y))
        # This is simpler and more efficient for SGD than R-style full least squares
        beta_init = jnp.zeros(n_features, dtype=X.dtype)
        if fit_intercept:
            # Compute starting mu from y mean (or distribution-specific starting value)
            mu_init = family.distribution.starting_mu(y)
            y_mean = jnp.mean(mu_init)
            # Compute intercept as link(y_mean)
            intercept_init = family.link.link(y_mean)
            # Handle offset: if offset is provided, adjust intercept
            if offset is not None:
                intercept_init = intercept_init - jnp.mean(offset)
            # Set intercept (last element in beta when fit_intercept=True)
            beta_init = beta_init.at[n_features - 1].set(intercept_init)

        beta_init = beta_init.reshape((-1, 1))

        # 3. Main optimization loop
        if is_early_stop_enabled:
            # State: (beta, converged, epoch)
            init_val = (beta_init, False, 0)

            def epoch_body(val):
                old_beta, _, epoch = val
                beta = sgd_epoch_update(
                    old_beta,
                    epoch,
                    xs,
                    ys,
                    offs,
                    sws,
                    family,
                    n_features,
                    l2,
                    fit_intercept,
                    learning_rate,
                    decay_rate,
                    decay_steps,
                )

                # Check convergence
                converged = check_convergence(beta, old_beta, stopping_rule, tol)
                converged = sml_reveal(converged)  # type: ignore

                return beta, converged, epoch + 1

            def cond_fun(val):
                _, converged, epoch = val
                return jnp.logical_and(epoch < max_iter, jnp.logical_not(converged))

            beta_final, converged, n_epochs = jax.lax.while_loop(
                cond_fun, epoch_body, init_val
            )
        else:
            # No early stopping: run fixed max_iter epochs
            def fixed_epoch_body(epoch, beta):
                return sgd_epoch_update(
                    beta,
                    epoch,
                    xs,
                    ys,
                    offs,
                    sws,
                    family,
                    n_features,
                    l2,
                    fit_intercept,
                    learning_rate,
                    decay_rate,
                    decay_steps,
                )

            beta_final = jax.lax.fori_loop(0, max_iter, fixed_epoch_body, beta_init)
            converged, n_epochs = False, max_iter

        # 4. Estimate Dispersion (not implemented yet)
        dispersion = None

        # 5. Build history (not implemented the full history yet)
        history = {
            "n_iter": n_epochs,
            "converged": converged,
        }

        if enable_spu_cache:
            X_train = sml_drop_cached_var(X_train)

        return beta_final, dispersion, history
