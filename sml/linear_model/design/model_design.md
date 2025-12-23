# model.py 设计细化（GLM Estimator）

目标：提供统一入口 `GLM`，完成 fit/predict/summary，封装 family/link/formula/solver/metrics 的组合与调度。

## 主要职责
- 组装 Family（含 canonical link 自动回退）。
- 选择 Formula（dispatcher：优先手写优化 -> 通用）。
- 选择 Solver（支持 'irls' 和 'sgd'）。
- 训练参数：`coef_`, `intercept_`, `dispersion_`（按分布需要）。
- 预测：`predict`（返回均值 `mu`），`predict_linear`（返回 `eta`）。
- 评估：`summary`（支持 deviance, aic, bic, auc 等）。

## 数据预处理 (Data Preprocessing)

### Y Scaling (Target Normalization)
在 MPC (如 SecretFlow SPU) 环境下，定点数计算的动态范围有限。对于 GLM，如果目标变量 $y$ 的数值范围较大（例如 Poisson/Gamma 回归中的赔款金额），直接计算会导致中间变量（如 `exp(eta)`）溢出或精度损失。

**最佳实践**：用户应手动对 $y$ 进行缩放，使其处于一个数值稳定的范围内（例如 $[0, 1]$ 或均值为 1 附近），并将该缩放因子 `scale` 传入 `fit` 和 `predict`。

**实现机制**:
1. `fit(X, y, scale=1.0)`: 内部使用 $y_{train} = y / scale$ 进行训练。
2. 模型学到的系数 $\beta$ 对应于缩放后的 $y_{train}$。
3. `predict(X, scale=1.0)`: 计算 $\mu_{scaled} = g^{-1}(X\beta)$，然后返回 $\mu = \mu_{scaled} * scale$。

**对系数的影响 (Impact on Coefficients)**:
根据 Link Function 的不同，Scaling 对系数的影响也不同：

1.  **Log Link (Tweedie, Gamma, Poisson)**:
    -   **斜率系数 (Slopes, $\beta_{i>0}$)**: **完全不变 (Invariant)**。缩放 $y$ 等价于 $\ln(y/K) = \ln(y) - \ln(K)$，平移项被截距吸收。这对于车险定价等关注相对风险因子 (Relativities) 的场景非常完美。
    -   **截距项 (Intercept, $\beta_0$)**: 发生平移。$\beta_0^{new} = \beta_0^{old} - \ln(scale)$。
    -   **正则化**: 由于斜率不变，通常**不需要**调整 L2 正则化强度。

2.  **Identity Link (Linear Regression, Gaussian)**:
    -   **所有系数**: 同比例缩小。$\beta^{new} = \beta^{old} / scale$。
    -   **正则化**: **必须**大幅减小 L2 正则化参数 (`l2`), 建议 $l2_{new} \approx l2_{old} / scale^2$，否则会导致严重欠拟合。

3.  **Logit Link (Binomial, Bernoulli)**:
    -   **不推荐缩放**。因为标签通常为 $\{0, 1\}$，无溢出风险。强行缩放会破坏 Sigmoid 函数的概率假设。

## 类接口
```python
class GLM:
    def __init__(
        self,
        dist,
        link=None,
        solver: str = "irls",
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 1e-2, 
        decay_rate: float = 1.0,     # New: SGD LR decay
        decay_steps: int = 100,      # New: SGD decay steps
        batch_size: int = 128,       
        l2: float = 0.0,
        fit_intercept: bool = True,
        formula=None,
        dispatcher=None,
        clip_eta: tuple | None = None,
        clip_mu: tuple | None = None,
    ):
        ...

    def fit(self, X, y, offset=None, sample_weight=None, scale: float = 1.0):
        """
        训练模型。
        
        Parameters:
        - offset: (n_samples,) 偏置项，固定加在线性预测子上。
        - scale: float, default=1.0. Target scaling factor for numerical stability in MPC.
                 Model fits on y / scale.
        """

    def predict(self, X, offset=None, scale: float = 1.0):
        """
        返回均值预测。
        mu = link.inverse(eta) * scale
        """

    def evaluate(self, X, y, metrics=("deviance", "aic", "rmse"), scale: float = 1.0):
        """计算多种评估指标。会自动处理 scale 反归一化。"""
```

## 数据流（fit）
1) 构造 `Family(dist, link)`。
2) 通过 dispatcher 选择 `Formula`。
3) 选择 `Solver`（'irls' 或 'sgd'）。
4) 调 solver：`solver.solve(..., learning_rate, decay_rate, decay_steps)`
   - 必须透传 SGD 特有的超参。
5) 拆分与保存结果。

## 关键参数处理
- `offset`: **Moved to fit()**. 这是一个数据依赖的参数，应与 X, y 同生命周期。
- `decay_rate` / `decay_steps`: 用于 SGD 的学习率调度。

## 异常与校验
- 校验 solver 名称。
- 校验 SGD 参数。
- 校验 X/y/offset 维度一致性。