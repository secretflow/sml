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


import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.family import Family


def deviance(
    y: jax.Array, mu: jax.Array, family: Family, sample_weight: jax.Array | None = None
) -> jax.Array:
    """Compute Deviance (lower is better)."""
    return family.distribution.deviance(y, mu, sample_weight)


def log_likelihood(
    y: jax.Array, mu: jax.Array, family: Family, sample_weight: jax.Array | None = None
) -> jax.Array:
    """Compute Log-Likelihood (higher is better)."""
    return family.distribution.log_likelihood(y, mu, sample_weight)


def aic(
    y: jax.Array,
    mu: jax.Array,
    family: Family,
    rank: int,
    sample_weight: jax.Array | None = None,
) -> jax.Array:
    """
    Compute Akaike Information Criterion (lower is better).
    AIC = -2 * LL + 2 * k
    """
    ll = log_likelihood(y, mu, family, sample_weight)
    return -2.0 * ll + 2.0 * rank


def bic(
    y: jax.Array,
    mu: jax.Array,
    family: Family,
    rank: int,
    n_samples: int,
    sample_weight: jax.Array | None = None,
) -> jax.Array:
    """
    Compute Bayesian Information Criterion (lower is better).
    BIC = -2 * LL + k * log(n)
    """
    ll = log_likelihood(y, mu, family, sample_weight)
    return -2.0 * ll + rank * jnp.log(n_samples)


def rmse(
    y: jax.Array, mu: jax.Array, sample_weight: jax.Array | None = None
) -> jax.Array:
    """
    Compute Root Mean Squared Error.
    """
    se = (y - mu) ** 2
    if sample_weight is not None:
        mse = jnp.sum(se * sample_weight) / jnp.sum(sample_weight)
    else:
        mse = jnp.mean(se)
    return jnp.sqrt(mse)


def auc(
    y: jax.Array, mu: jax.Array, sample_weight: jax.Array | None = None
) -> jax.Array:
    """
    Compute Area Under the ROC Curve.
    
    WARNING: This requires sorting, which involves significant overhead in MPC environments.
    """
    # Simple implementation using trapezoidal rule on sorted data
    # Note: scikit-learn's roc_auc_score is more complex.
    # Here we implement a basic version.
    
    # 1. Sort by prediction (descending)
    desc_score_indices = jnp.argsort(mu)[::-1]
    y_sorted = y[desc_score_indices]
    
    if sample_weight is not None:
        weight_sorted = sample_weight[desc_score_indices]
    else:
        weight_sorted = jnp.ones_like(y)

    # 2. Compute TPR and FPR
    # distinct_value_indices = jnp.where(jnp.diff(mu_sorted))[0]
    # threshold_idxs = jnp.r_[distinct_value_indices, y_sorted.size - 1]
    
    # Simplified: Assume all scores distinct or just integrate
    # TPS = cumsum(y * w), FPS = cumsum((1-y) * w)
    tps = jnp.cumsum(y_sorted * weight_sorted)
    fps = jnp.cumsum((1 - y_sorted) * weight_sorted)
    
    # 3. Trapezoidal rule
    # AUC = sum of (fpr_i - fpr_{i-1}) * tpr_i
    # We normalized TPR/FPR by total positives/negatives at the end
    total_tp = tps[-1]
    total_fp = fps[-1]
    
    # Avoid division by zero
    total_tp = jnp.where(total_tp == 0, 1.0, total_tp)
    total_fp = jnp.where(total_fp == 0, 1.0, total_fp)

    tpr = tps / total_tp
    fpr = fps / total_fp
    
    # Append (0,0)
    tpr = jnp.concatenate([jnp.array([0.0]), tpr])
    fpr = jnp.concatenate([jnp.array([0.0]), fpr])
    
    return jnp.trapz(tpr, fpr)
