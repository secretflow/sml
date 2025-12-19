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

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.formula.base import Formula

from .base import Solver
from .utils import add_intercept


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

    Supports Mini-Batch SGD via sequential slicing.
    """

    def solve(
        self,
        X: jax.Array,
        y: jax.Array,
        family: Family,
        formula: Formula,
        fit_intercept: bool = True,
        offset: Optional[jax.Array] = None,
        sample_weight: Optional[jax.Array] = None,
        l2: float = 0.0,
        max_iter: int = 100,  # Interpreted as Epochs
        tol: float = 1e-4,
        learning_rate: float = 1e-2,
        decay_rate: float = 1.0,      # New: LR decay
        decay_steps: int = 100,       # New: LR decay steps
        batch_size: int = 128,
        random_state: Optional[int] = None, # Used for Shuffle if implemented (future)
        clip_eta: Optional[Tuple[float, float]] = None,
        clip_mu: Optional[Tuple[float, float]] = None,
    ) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        # Handle batch size
        batch_size = min(batch_size, n_samples)
        n_batches = n_samples // batch_size
        
        # 2. Initialization
        beta_init = jnp.zeros(n_features, dtype=X.dtype)

        # State: (beta, converged, epoch, deviance)
        init_val = (beta_init, False, 0, jnp.inf)

        # 3. Define Batch Step function
        def batch_step(batch_idx, val):
            beta, epoch = val
            
            # Calculate current step (global) for decay
            global_step = epoch * n_batches + batch_idx
            
            # Decay learning rate
            # lr = base_lr * decay_rate ^ (step / decay_steps)
            decay_factor = jnp.power(decay_rate, global_step / decay_steps)
            current_lr = learning_rate * decay_factor

            start = batch_idx * batch_size
            # Use dynamic_slice for JIT compatibility
            X_batch = jax.lax.dynamic_slice(
                X_train, (start, 0), (batch_size, n_features)
            )
            y_batch = jax.lax.dynamic_slice(y, (start,), (batch_size,))

            off_batch = None
            if offset is not None:
                off_batch = jax.lax.dynamic_slice(offset, (start,), (batch_size,))

            sw_batch = None
            if sample_weight is not None:
                sw_batch = jax.lax.dynamic_slice(sample_weight, (start,), (batch_size,))

            # Compute components on batch
            w, z_resid, _, _, _, _ = formula.compute_components(
                X=X_batch,
                y=y_batch,
                beta=beta,
                offset=off_batch,
                family=family,
                sample_weight=sw_batch,
                clip_eta=clip_eta,
                clip_mu=clip_mu,
            )

            # Gradient Calculation
            # Gradient = X^T * (W * z_resid)
            grad_components = w * z_resid
            grad = X_batch.T @ grad_components

            # L2 Regularization
            if l2 > 0:
                l2_grad = l2 * beta
                if fit_intercept:
                    l2_grad = l2_grad.at[n_features - 1].set(0.0)
                grad -= l2_grad

            # Update
            beta_new = beta + current_lr * grad
            return (beta_new, epoch)

        # 4. Define Epoch Loop (iterates over batches)
        def epoch_body(val):
            beta, _, epoch, _ = val
            
            # Loop over batches
            # carry is (beta, epoch) because we need epoch for global step calculation
            beta_next, _ = jax.lax.fori_loop(
                0, n_batches, batch_step, (beta, epoch)
            )
            
            # Calculate full deviance for convergence check
            _, _, _, _, dev_full, _ = formula.compute_components(
                X=X_train,
                y=y,
                beta=beta_next,
                offset=offset,
                family=family,
                sample_weight=sample_weight,
                clip_eta=clip_eta,
                clip_mu=clip_mu,
            )

            beta_diff = jnp.linalg.norm(beta_next - beta)
            beta_norm = jnp.linalg.norm(beta)
            rel_change = beta_diff / (beta_norm + 1e-12)
            converged = rel_change < tol
            
            return beta_next, converged, epoch + 1, dev_full

        # 5. Define Cond function
        def cond_fun(val):
            _, converged, epoch, _ = val
            return jnp.logical_and(epoch < max_iter, jnp.logical_not(converged))

        # 6. Run
        beta_final, converged, n_epochs, dev_final = jax.lax.while_loop(
            cond_fun, epoch_body, init_val
        )

        # 7. Dispersion
        dispersion = dev_final / (n_samples - n_features)

        history = {
            "n_iter": n_epochs,
            "converged": converged,
            "final_deviance": dev_final,
        }

        return beta_final, dispersion, history
