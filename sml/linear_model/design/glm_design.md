# JAX-GLM 设计方案（解耦、可扩展、手写优化友好）

本文档描述在 JAX 上实现广义线性模型（GLM）的架构设计，目标是：
- **充分解耦**：链接函数（Link）、分布（Distribution）、公式策略（Formula）、求解器（Solver）、评估指标（Metrics）彼此独立且可插拔。
- **默认友好**：每个分布都绑定默认的 Canonical Link，小白用户零配置可用。
- **双路径计算**：通用数学公式路径 + 针对特定组合的手写优化路径，可无缝切换。
- **显式数学**：不依赖 `jax.grad`/`jax.hessian`/`vmap`，核心使用显式推导与基础线性代数算子。
- **特定后端适配**：为适配 MPC 等后端，避免使用 `solve` 等复杂分解算子，强制使用 **Naive Matrix Inversion**。

## 总览分层
```
sml/linear_model/glm/
├── model.py          # GLM Estimator：fit/predict/evaluate
├── core/             # 数学原子：Link、Distribution、Family
├── formula/          # 公式策略：通用 Generic + 手写 Optimized；分发器 Dispatcher
├── solvers/          # 求解器：IRLS (Naive Inv), SGD
└── metrics/          # Deviance, AIC, BIC, RMSE, AUC
```

推荐目录（按你当前仓库布局，放在 `sml/linear_model` 下）：
```
sml/linear_model/
├── glm/
│   ├── __init__.py
│   ├── model.py            # 用户入口
│   ├── core/               # Link, Distribution, Family
│   ├── formula/            # Generic, Optimized, Dispatcher
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py         # Solver Protocol
│   │   ├── irls.py         # IRLS (Naive Inversion + Jitter)
│   │   ├── sgd.py          # SGD (Gradient Ascent/Descent)
│   │   └── utils.py        # invert_matrix, add_intercept 等
│   └── metrics/
│       ├── __init__.py
│       └── metrics.py      # deviance, aic, bic, rmse, auc
└── design/
    └── glm_design.md       # 本设计文档
```

## 核心设计要点回顾
1.  **Naive Matrix Inversion**:
    - 所有涉及线性方程组求解的地方（主要是 IRLS 中的 Update），**必须** 使用 `inv(H) @ z` 的形式。
    - **禁止** 使用 `solve`, `cholesky`, `qr`。
    - 必须在求逆前对矩阵对角线添加 `epsilon` 保证可逆。

2.  **SGD Solver**:
    - 新增随机梯度下降求解器。
    - 支持 Batch 训练和 **Learning Rate Decay**。
    - 利用 `Formula` 计算梯度分量 $W \cdot z_{resid}$，避免重复造轮子。

3.  **Metrics**:
    - 扩展支持 AIC, BIC, RMSE, AUC。
    - 注意 AUC 在 MPC 环境下的排序开销。

## 示例：注册 Tweedie+Log 手写公式
```python
from sml.linear_model.glm.formula.dispatch import register_formula
from sml.linear_model.glm.formula.optimized import TweedieLogFormula
from sml.linear_model.glm.core.distribution import Tweedie
from sml.linear_model.glm.core.link import LogLink

register_formula(dist_type=Tweedie, link_type=LogLink, formula_cls=TweedieLogFormula)

model = GLM(dist=Tweedie(p=1.5), solver='sgd', learning_rate=0.01, decay_rate=0.95)
model.fit(X, y)
```
