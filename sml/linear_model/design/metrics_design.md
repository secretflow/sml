# metrics 设计细化（Deviance & More）

提供丰富的评估指标，包括 Deviance, AIC, BIC, RMSE, AUC 等。

## 接口
```python
def deviance(y, mu, family, sample_weight=None):
    """Deviance (lower is better)."""

def log_likelihood(y, mu, family, sample_weight=None):
    """Log-Likelihood (higher is better)."""

def aic(y, mu, family, rank, sample_weight=None):
    """
    Akaike Information Criterion.
    AIC = -2 * LL + 2 * rank
    """

def bic(y, mu, family, rank, n_samples, sample_weight=None):
    """
    Bayesian Information Criterion.
    BIC = -2 * LL + rank * log(n_samples)
    """

def rmse(y, mu, sample_weight=None):
    """
    Root Mean Squared Error.
    sqrt(mean((y - mu)^2))
    """

def auc(y, mu, sample_weight=None):
    """
    Area Under the ROC Curve (for binary classification).
    Requires sorting predictions, which is expensive in MPC.
    """
```

## 实现要点

### AIC / BIC
- 依赖 `family.distribution.log_likelihood`。
- `rank` 等于模型自由度（特征数 + intercept）。
- `n_samples` 为样本数量。

### RMSE
- 通用回归指标。
- `mse = average((y - mu)**2, weights=sample_weight)`
- `rmse = sqrt(mse)`

### AUC (ROC)
- **MPC 挑战**：计算 AUC 需要对预测值 `mu` 进行排序（`O(N log N)`），在 MPC（尤其是 SPU）中通过 Oblivious Sort 实现，开销极大。
- **设计**：
  - 提供标准实现，基于 `jax.numpy.argsort`。
  - **文档警告**：明确标注在大数据量 MPC 场景下的性能风险，建议仅在小数据或明文通过 `reveal` 后计算。

## 扩展点
- **Confusion Matrix**: 需要阈值切分。
- **R2 Score**: 传统的 $1 - SS_{res} / SS_{tot}$。

## 依赖边界
- 指标函数保持纯函数签名。