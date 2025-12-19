# model.py 设计细化（GLM Estimator）

目标：提供统一入口 `GLM`，完成 fit/predict/summary，封装 family/link/formula/solver/metrics 的组合与调度。

## 主要职责
- 组装 Family（含 canonical link 自动回退）。
- 选择 Formula（dispatcher：优先手写优化 -> 通用）。
- 选择 Solver（支持 'irls' 和 'sgd'）。
- 训练参数：`coef_`, `intercept_`, `dispersion_`（按分布需要）。
- 预测：`predict`（返回均值 `mu`），`predict_linear`（返回 `eta`）。
- 评估：`summary`（支持 deviance, aic, bic, auc 等）。

## 类接口（示例）
```python
class GLM:
    def __init__(
        self,
        dist,
        link=None,
        solver: str = "irls",
        formula=None,
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 1e-2, # New: for SGD
        batch_size: int = 128,       # New: for SGD
        l2: float = 0.0,
        fit_intercept: bool = True,
        offset=None,
        sample_weight=None,
        dispatcher=None,
        clip_eta: tuple | None = None,
        clip_mu: tuple | None = None,
    ):
        ...

    def fit(self, X, y):
        """训练模型，保存 coef_, intercept_, dispersion_。"""

    def predict(self, X, offset=None):
        """返回均值预测 mu = link.inverse(eta)。"""

    def score(self, X, y):
        """默认返回负 deviance。"""
    
    def evaluate(self, X, y, metrics=("deviance", "aic", "rmse")):
        """计算多种评估指标。"""
```

## 数据流（fit）
1) 构造 `Family(dist, link)`。
2) 通过 dispatcher 选择 `Formula`。
3) 选择 `Solver`（'irls' 或 'sgd'）。
4) 调 solver：`solver.solve(..., learning_rate, batch_size)`
   - 必须透传 SGD 特有的超参。
   - Solver 内部必须遵循 **Naive Matrix Inversion** 约束（针对 IRLS）。
5) 拆分与保存结果。

## 关键参数处理
- `solver`: 字符串，'irls' (默认) 或 'sgd'。
- `learning_rate` / `batch_size`: 仅当 `solver='sgd'` 时生效。
- `fit_intercept`: 训练时在 X 右侧追加常数列；推理时自动加上 intercept。

## 异常与校验
- 校验 solver 名称。
- 校验 SGD 参数（lr > 0, batch_size > 0）。
- **Naive Inversion Warning**: 提示用户当前实现使用显式求逆，可能存在数值稳定性风险（尽管已加 jitter）。

## 扩展点
- 支持更多 solver（保持 solve 接口一致）。
- 支持更多 metrics（summary 透传到 metrics 模块）。
