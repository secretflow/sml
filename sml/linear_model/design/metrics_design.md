# metrics 设计细化（Deviance）

当前仅提供 deviance 指标，保持独立、无副作用。

## 接口
```python
def deviance(y, mu, family, sample_weight=None):
    """
    计算 deviance，委托到 family.distribution.deviance。
    返回标量。
    """
```

## 实现要点
- 调用 `family.distribution.deviance(mu, y, weights=sample_weight)`（注意参数顺序保持与分布实现一致）。
- 输入校验：`y.shape == mu.shape`；sample_weight 若给定，与 y 同形或可广播。
- 不在 metrics 内部做裁剪，假设上游 predict 已处理 `mu` 稳定性。

## 扩展点
- 后续可追加：AIC, BIC, pseudo-R², logloss, MSE/MAE, KS/ROC/AUC 等。
- 所有指标函数应保持纯函数签名，避免依赖 solver 状态。