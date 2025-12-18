# metrics 设计细化（Deviance & More）

当前主要提供 deviance 指标，保持独立、无副作用。后续支持 AIC/BIC。

## 接口
```python
def deviance(y, mu, family, sample_weight=None):
    """
    计算 deviance，委托到 family.distribution.deviance。
    返回标量。
    """

def log_likelihood(y, mu, family, sample_weight=None):
    """
    计算对数似然，委托到 family.distribution.log_likelihood。
    返回标量。
    """
```

## 实现要点
- 调用 `family.distribution.deviance` 或 `log_likelihood`。
- 输入校验：`y.shape == mu.shape`；sample_weight 若给定，与 y 同形或可广播。
- 不在 metrics 内部做裁剪，假设上游 predict 已处理 `mu` 稳定性。

## 扩展点 (AIC / BIC)
AIC 和 BIC 依赖于 Log-Likelihood。虽然 $Deviance \approx -2 LL + C$，但为了精确计算，我们优先使用显式的 `log_likelihood`。
- `AIC = -2 * LL + 2 * k`
- `BIC = -2 * LL + k * ln(n)`
- 其中 `k` 为模型参数个数 (df_model + 1)，`n` 为样本数。

## 依赖边界
- 所有指标函数应保持纯函数签名，避免依赖 solver 状态。
