# JAX-GLM 设计方案（解耦、可扩展、手写优化友好）

本文档描述在 JAX 上实现广义线性模型（GLM）的架构设计，目标是：
- **充分解耦**：链接函数（Link）、分布（Distribution）、公式策略（Formula）、求解器（Solver）、评估指标（Metrics）彼此独立且可插拔。
- **默认友好**：每个分布都绑定默认的 Canonical Link，小白用户零配置可用。
- **双路径计算**：通用数学公式路径 + 针对特定组合的手写优化路径，可无缝切换。
- **显式数学**：不依赖 `jax.grad`/`jax.hessian`/`vmap`，核心使用显式推导与基础线性代数算子，保证数值稳定与可解释性。
- **易扩展**：便于添加新 Link/Distribution/Solver/Formula/Metric，以及注册业务定制优化。

## 总览分层
```
sml/linear_model/glm/
├── model.py          # GLM Estimator：fit/predict/summary（原“api.py”的职责）
├── core/             # 数学原子：Link、Distribution、Family
├── formula/          # 公式策略：通用 Generic + 手写 Optimized；分发器 Dispatcher
├── solvers/          # IRLS / Newton（显式推导路径）
└── metrics/          # Deviance, AIC, BIC, pseudo-R², KS/ROC(二分类)
```

推荐目录（按你当前仓库布局，放在 `sml/linear_model` 下）：
```
sml/linear_model/
├── glm/                    # 新的 JAX-GLM 实现放在这里
│   ├── __init__.py
│   ├── model.py            # 用户入口：GLM 类（fit/predict/summary）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── link.py         # Link 基类与内置实现：Identity, Log, Logit, Probit, Reciprocal, Power, CLogLog
│   │   ├── distribution.py # 分布基类与实现：Gaussian, Bernoulli, Poisson, Gamma, Tweedie, NegBin
│   │   └── family.py       # Family 容器：绑定 dist + link，提供 canonical fallback
│   ├── formula/
│   │   ├── __init__.py
│   │   ├── base.py         # Formula 协议，定义 compute_components 接口
│   │   ├── generic.py      # 通用公式（支持 IRLS/Newton）
│   │   ├── optimized.py    # 手写优化公式集合（如 Tweedie+Log, Gamma+Log）
│   │   └── dispatch.py     # Dispatcher/Registry，按 (dist, link) 选择公式
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── irls.py         # IRLS 主流程，显式 W/z 计算
│   │   ├── newton.py       # Newton-Raphson 路径（显式一阶/二阶）
│   │   └── utils.py        # 收敛判定、阻尼/line-search、正则项拼装
│   └── metrics/
│       ├── __init__.py
│       └── summary.py      # deviance, AIC, BIC, pseudo-R², KS/ROC/PR
└── design/
    └── glm_design.md       # 本设计文档
```

## 核心原子层：Link / Distribution / Family
### Link
职责：完成 `mu ↔ η` 变换及导数，支持 IRLS/Newton 需要的 `g'(mu)` 与 `h'(eta)`。

接口契约：
- `link(mu) -> eta`：$\eta = g(\mu)$
- `inverse(eta) -> mu`：$\mu = g^{-1}(\eta)$
- `link_deriv(mu) -> d_eta/d_mu`：用于 IRLS 的工作响应
- `inverse_deriv(eta) -> d_mu/d_eta`：用于 Newton 的链式法则

实现建议：Identity, Log, Logit, Probit, CLogLog, Power(a), Reciprocal 等。

### Distribution
职责：提供方差函数、deviance 计算、起始均值等统计属性。

接口契约：
- `unit_variance(mu)` 返回 $V(\mu)$。
- `deviance(y, mu, weights=None)` 用于评估，不参与梯度。
- `starting_mu(y)` 给出稳健初值，避免 log(0)。
- `get_canonical_link()` 返回 canonical link；若无（如 Tweedie），约定默认 Log。

### Family
容器：将分布与链接绑定，自动回退 canonical link。
```python
class Family:
    def __init__(self, distribution, link=None):
        self.distribution = distribution
        self.link = link or distribution.get_canonical_link()
```

## 公式策略层：Generic 与 Hand-Optimized
核心思想：把“如何计算工作权重 W 与工作残差 z_resid”的逻辑从 solver 中剥离。solver 只消费 `(W, z_resid, mu, eta)`。

接口：
```python
class Formula(Protocol):
    def compute_components(self, X, y, beta, offset, family):
        """
        返回:
        - weights W: (N,)
        - z_resid: (y - mu) * g'(mu)
        - mu, eta: 便于 solver 复用
        - extras: 可选调试信息（如对数似然）
        """
```

### 通用公式 Generic
- 教科书 IRLS：
  - $W = 1 / (V(\mu) \cdot (g'(\mu))^2)$
  - $z_{resid} = (y - \mu) \cdot g'(\mu)$
- 支持数值裁剪：`mu`/`eta` clip，`W` 上下界防溢出/奇异。

### 手写优化公式 Optimized（可注册多组）
典型示例：
- Tweedie + Log：`mu = exp(eta)`, `W = mu**(2-p)`, `z_resid = y/mu - 1`。
- Gamma + Log：`W = 1`, `z_resid = (y - mu) / mu`。
- Poisson + Log：`W = mu`, `z_resid = (y - mu) / mu`。

手写公式允许你：
- 用化简后的闭式表达式避免中间项爆炸（例如先算 $g'(\mu)$ 再平方再取倒数）。
- 加入领域特定的裁剪/重参数化。

### Dispatcher / Registry
- 维护映射 `(dist_type, link_type) -> Formula`。
- 查找顺序：用户自定义 > 内置 optimized > 默认 generic。
- 对外暴露 `register_formula(dist, link, formula_cls)` 便于业务方注入。

## 求解器层（仅迭代与更新规则，不做业务逻辑）
### IRLS
输入：`X, y, family, formula, offset, sample_weight, l2, max_iter, tol`。

迭代：
1. `eta = X @ beta + offset`
2. 调用 Formula 得到 `W, z_resid, mu, eta`。
3. `z = eta + z_resid`
4. 组装 `H = X^T W X (+ l2 I)`，`score = X^T W z`
5. 计算更新 `beta_new`（实现上可用 `jnp.linalg.solve` 等基础线性代数原语）。
6. 收敛判定：`||beta_new - beta|| / (||beta||+eps) < tol` 或 deviance 改善。

支持：偏置列自动填充、offset、sample_weight、L2/elastic-net（对 bias 屏蔽正则）。

### Newton-Raphson（显式一二阶）
- $score = X^T (y - \mu) / Var(\mu) \cdot h'(\eta)$
- $H = X^T diag(h'(\eta)^2 / Var(\mu)) X (+ l2 I)$
- 可选阻尼 / 线搜索，保障收敛。

### 数值要点
- 不使用 `vmap/grad/hessian`，所有导数与权重均为显式公式。
- 数据很大时：可以在 solver 内部做简单的批处理计算（仍基于显式导数/公式），但不引入额外高阶算子。

## Metrics 层
- `deviance`, `AIC`, `BIC`, `pseudo R² (McFadden/Cox-Snell/Nagelkerke)`, `logloss`, `MSE/MAE`，分类可选 `KS/ROC/AUC`。
- 仅消费 `y, mu/pred, weights, dispersion`，独立于 solver。

## API 层：GLM Estimator（`model.py`）
- 入口（示意）：`GLM(dist, link=None, solver='irls', formula=None, max_iter=100, tol=1e-6, l2=0, offset=None, fit_intercept=True, sample_weight=None)`。
- `fit(X, y)`：
  - 构建 Family（含 canonical link 回退）
  - 通过 Dispatcher 选择 formula（optimized 或 generic）
  - 运行 solver，保存 `coef_, intercept_, dispersion_`
- `predict(X, offset=None)`：`eta = X@coef + intercept + offset`，经 `link.inverse(eta)` 得到 `mu`。
- `summary(X, y, metrics=[...])`：计算指标、返回收敛信息。

## 数据流（以 IRLS 为例）
```
User -> GLM.fit
  -> Family(link fallback)
  -> FormulaDispatcher(select optimized or generic)
  -> Solver(IRLS)
      loop:
        eta = X@beta + offset
        W, z_resid, mu = formula.compute(...)
        z = eta + z_resid
        beta = update( X^T W X , X^T W z )
      -> convergence
  -> store coef_/intercept_
```

## 扩展指引
1. **新增分布**：实现 `Distribution` 接口 + `get_canonical_link`；注册到 `Family` 工厂。
2. **新增 link**：实现 `Link` 接口；若为某分布的 canonical，更新分布的 `get_canonical_link`。
3. **新增手写优化**：实现 `Formula`，注册 `(dist_type, link_type)` 到 `Dispatcher`。
4. **新增 solver**：实现统一签名 `solve(X, y, family, formula, **kwargs)`；仅依赖导数/公式产物。
5. **新增指标**：在 `metrics.summary` 增加函数即可，完全独立。

## 数值稳定策略
- `mu`、`eta` 裁剪：避免 `log(0)`, `exp` 溢出（例如 clamp 到 `[eps, upper]`）。
- `W` 上下界：`W = clip(W, w_min, w_max)`，防止权重过大导致病态。
- `starting_mu`：分布内提供稳健初值（如 `(y + mean(y))/2`）。
- Newton 可提供阻尼（damping）与最小步长限制。

## 示例：注册 Tweedie+Log 手写公式
```python
from sml.linear_model.glm.formula.dispatch import register_formula
from sml.linear_model.glm.formula.optimized import TweedieLogFormula
from sml.linear_model.glm.core.distribution import Tweedie
from sml.linear_model.glm.core.link import LogLink

register_formula(dist_type=Tweedie, link_type=LogLink, formula_cls=TweedieLogFormula)

model = GLM(dist=Tweedie(p=1.5))  # 未传 link，自动使用 canonical Log
model.fit(X, y)
```

## 后续可选增强
- 交叉验证/正则路径（lambda path）。
- 自动超参建议：根据分布/样本量推荐 `max_iter`、`tol`、`ridge`。
- 更丰富的业务特定公式库（保险、风控领域常见组合）。
