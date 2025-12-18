# model.py 设计细化（GLM Estimator）

目标：提供统一入口 `GLM`，完成 fit/predict/summary，封装 family/link/formula/solver/metrics 的组合与调度。

## 主要职责
- 组装 Family（含 canonical link 自动回退）。
- 选择 Formula（dispatcher：优先手写优化 -> 通用）。
- 选择 Solver（默认 IRLS，可选 Newton）。
- 训练参数：`coef_`, `intercept_`, `dispersion_`（按分布需要）。
- 预测：`predict`（返回均值 `mu`），`predict_linear`（返回 `eta`）。
- 评估：`summary`（至少提供 deviance）。

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
        tol: float = 1e-6,
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
        """训练模型，保存 coef_, intercept_, dispersion_（视分布需要）。"""

    def predict(self, X, offset=None):
        """返回均值预测 mu = link.inverse(eta)。"""

    def predict_linear(self, X, offset=None):
        """返回线性预测子 eta。"""

    def summary(self, X, y, metrics=("deviance",)):
        """计算指标，当前至少支持 deviance。"""
```

## 数据流（fit）
1) 构造 `Family(dist, link)`，若 link 为空则用分布的 canonical link。
2) 通过 dispatcher 选择 formula（若用户显式传入则直接用）。
3) 选择 solver（"irls" / "newton"）。
4) 调 solver：`solver.solve(X, y, family, formula, **kwargs)`
   - solver 返回 `beta`, `dispersion`, `history`（可选）。
5) 拆分 `beta` 为 `coef_` 与 `intercept_`（若 fit_intercept=True）。

## 关键参数处理
- `fit_intercept`: 训练时在 X 右侧追加常数列；推理时自动加上 intercept。正则时对 intercept 屏蔽。
- `offset`: 训练与预测均可传入，按样本加到 `eta`。
- `sample_weight`: 透传给 formula/solver，用于 W 或残差加权。
- `clip_eta`, `clip_mu`: 数值稳定裁剪，传递给 formula 用于 `eta`/`mu` clip。
- `l2`: 传给 solver，构建对角抬升矩阵（对 intercept 置 0）。

## 伪代码（fit）
```python
def fit(self, X, y):
    family = Family(self.dist, self.link)
    formula = self.formula or dispatcher.resolve(family.distribution, family.link)
    solver = get_solver(self.solver)
    beta, dispersion, history = solver.solve(
        X=X,
        y=y,
        family=family,
        formula=formula,
        fit_intercept=self.fit_intercept,
        offset=self.offset,
        sample_weight=self.sample_weight,
        l2=self.l2,
        max_iter=self.max_iter,
        tol=self.tol,
        clip_eta=self.clip_eta,
        clip_mu=self.clip_mu,
    )
    self.coef_, self.intercept_ = split_coef(beta, self.fit_intercept)
    self.dispersion_ = dispersion
    self.history_ = history
    return self
```

## predict / predict_linear
```python
def predict_linear(self, X, offset=None):
    eta = X @ coef + intercept
    if offset is not None: eta += offset
    return eta

def predict(self, X, offset=None):
    eta = predict_linear(...)
    return family.link.inverse(eta)
```

## summary
- 输入：`X, y, metrics=("deviance",)`
- 流程：先 `mu = predict(X, offset)`，再调用 `metrics.deviance(y, mu, family, sample_weight)`。
- 输出：dict，例如 `{ "deviance": value, "n_iter": history.n_iter, "converged": bool }`。

## 异常与校验
- 校验 solver 名称与 formula 类型。
- 校验 X/y 形状、样本数一致；sample_weight/offset 维度对齐。
- 对数值裁剪范围进行 sanity check（下界 < 上界）。

## 扩展点
- 支持更多 solver（保持 solve 接口一致）。
- 支持更多 metrics（summary 透传到 metrics 模块）。
- dispatcher 可由用户注册自定义公式。