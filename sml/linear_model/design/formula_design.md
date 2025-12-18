# formula 设计细化（Generic / Optimized / Dispatcher）

核心思想：把“W 与 z_resid 的计算”从 solver 中剥离。solver 只消费 `(W, z_resid, mu, eta)`。

## 接口
```python
class Formula(Protocol):
    def compute_components(self, X, y, beta, offset, family, sample_weight=None, clip_eta=None, clip_mu=None):
        """
        返回:
        - W: (N,) 工作权重
        - z_resid: (N,) 工作残差 (y - mu) * g'(mu)
        - mu: 期望
        - eta: 线性预测子
        - deviance: (float) 当前点的 deviance (用于 Line Search 判定)
        - extras: 可选字典（如 loglike）
        """
```

## GenericFormula
- 步骤：
  1. `eta = X @ beta + offset`
  2. `eta = clip_eta(eta)`（可选）
  3. `mu = family.link.inverse(eta)`；`mu = clip_mu(mu)`（可选）
  4. `v = family.distribution.unit_variance(mu)`
  5. `g_prime = family.link.link_deriv(mu)`
  6. `W = 1 / (v * g_prime**2)`，可 clip 到 `[w_min, w_max]`
  7. `z_resid = (y - mu) * g_prime`
  8. `dev = family.distribution.deviance(y, mu, sample_weight)`
  9. 若 `sample_weight` 给定，则 `W *= sample_weight`

- 数值要点：
  - 对 `eta`、`mu`、`W` 做裁剪，防止溢出或奇异。
  - `g_prime` 避免为 0，可加 eps。
  - 顺手计算 `dev` 比在外部单独计算更高效（mu 已就绪）。

## OptimizedFormula（示例）
- Tweedie + Log：
  - `eta = X@beta + offset`；`mu = exp(eta)`
  - `W = mu**(2-p)`
  - `z_resid = y/mu - 1`
  - `dev`：使用优化后的 deviance 公式计算。
- Gamma + Log：
  - `mu = exp(eta)`
  - `W = 1`
  - `z_resid = (y - mu) / mu`
- Poisson + Log：
  - `mu = exp(eta)`
  - `W = mu`
  - `z_resid = (y - mu) / mu`

优化公式可携带：
- 更强的裁剪策略（对 eta 设上限防 exp 爆炸）。
- 直接写闭式，跳过中间导数的除法，提升稳定性。

## Dispatcher / Registry
- 维护映射 `(dist_type, link_type) -> Formula`。
- 查找顺序：用户注册 > 内置 optimized > Generic。

API 示例：
```python
register_formula(dist_type, link_type, formula_cls)
resolve(distribution, link) -> formula_instance
```

## 依赖边界
- 不调用 `jax.grad` 等高阶算子。
- 仅使用显式数学与基础 `jax.numpy` 运算。

## 扩展指引
- 新增优化公式：实现 `compute_components` 并注册。
- 如需额外输出，放入 `extras` 字典，solver 可忽略或记录。
