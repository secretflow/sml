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
在 MPC (如 SecretFlow SPU) 环境下，定点数计算的动态范围有限。对于 GLM，如果目标变量 $y$ 的数值范围较大（例如 Poisson 回归中的计数），直接计算会导致中间变量（如 `exp(eta)`）溢出或精度损失。

**最佳实践**：用户应手动对 $y$ 进行缩放，使其处于一个数值稳定的范围内（例如 $[0, 1]$ 或均值为 1 附近），并将该缩放因子 `scale` 传入 `fit` 和 `predict`。

示例：
```python
# 手动计算 scale
y_scale = y.max() / 2.0
if y_scale < 1.0:
    y_scale = 1.0

# 传入 scale 进行训练
model.fit(X, y, scale=y_scale)

# 预测时传入 scale 进行反归一化
mu = model.predict(X, scale=y_scale)
```

**实现机制**:
1. `fit(X, y, scale=1.0)`: 内部使用 $y_{train} = y / scale$ 进行训练。
2. 模型学到的系数 $\beta$ 对应于缩放后的 $y_{train}$。
3. `predict(X, scale=1.0)`: 计算 $\mu_{scaled} = g^{-1}(X\beta)$，然后返回 $\mu = \mu_{scaled} * scale$。

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
        random_state: int | None = None, # New: JAX PRNGKey seed
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
- `random_state`: 用于初始化 JAX 的随机数生成器（如 SGD 初始化或 Shuffle）。

## 异常与校验
- 校验 solver 名称。
- 校验 SGD 参数。
- 校验 X/y/offset 维度一致性。