# solvers 设计细化（IRLS / Newton-CG）

目标：提供显式迭代求解器，不依赖 autograd/vmap/hessian。仅使用基础线性代数原语。

## 统一接口
```python
class Solver(Protocol):
    def solve(
        self,
        X,
        y,
        family,
        formula,
        fit_intercept: bool = True,
        offset=None,
        sample_weight=None,
        l2: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        clip_eta=None,
        clip_mu=None,
    ) -> tuple[beta, dispersion, history]:
        ...
```
- `beta`: (p or p+1,) 含 intercept。
- `dispersion`: 对需要尺度参数的分布（如 Gaussian/Tweedie）可估计；否则为 None 或 1。
- `history`: 可包含 `n_iter`, `converged`, `obj_trace` 等。

## IRLS (Iterative Reweighted Least Squares)
核心步骤：
1. 初始化 `beta = 0`，可用 `family.distribution.starting_mu(y)` 推断初值。
2. 迭代直到收敛或 `max_iter`：
   - `eta = X @ beta + offset`
   - 调 `formula.compute_components(...)` 得到 `W, z_resid, mu, eta, dev`
   - `z = eta + z_resid`
   - 构造带权矩阵：`Xw = sqrt(W) * X`，`zw = sqrt(W) * z`
   - 正则：对除 intercept 以外的列加 `l2` 对角抬升
   - 更新：`beta_new = solve( Xw^T Xw + l2I , Xw^T zw )`（用 `jnp.linalg.solve` 或 Cholesky）
   - 收敛：`max(|beta_new - beta|) / (max(|beta|)+eps) < tol`
3. dispersion 估计：
   - 对指数族常见做法：`phi = deviance / dof_resid`（可选，按分布决定是否需要）。

数值要点：
- 对 `W` 设下界 `w_min`，避免奇异矩阵。
- 拆分 intercept 正则：在对角抬升时对 intercept 位置置 0。

## Newton-CG / Fisher Scoring
GLM 中的 "Newton" 通常指代 **Fisher Scoring**，它使用 **期望 Hessian** (Expected Hessian) 而非观测 Hessian。
- **优点**：期望 Hessian 恰好等于 $X^T W X$，这使得 Fisher Scoring 与 IRLS 共享完全相同的 `W` 计算逻辑。
- **差异**：本 Solver 实现相比 IRLS 增加了 **Line Search** 和更灵活的线性方程求解（如 CG），能更好地处理病态问题或非凸区域。

流程：
- `score = X^T * (y - mu) * link_deriv(mu) / V(mu)` (即梯度)
- `H = X^T * W * X + l2 * I` (使用 Formula 计算的 W，即期望 Hessian)
- 更新方向：`delta = solve(H, score)`
- **Line Search**：检查 `formula` 返回的 `deviance` 是否下降。若未下降，缩小步长。

数值要点：
- 默认使用 Fisher Scoring (Expected Hessian) 以保证稳健性。
- 若需 Exact Newton (Observed Hessian)，需扩展 Formula 以返回二阶导数项 (包含 $y-\mu$ 项)，目前暂不作为默认路径。

## 工具函数（solvers/utils.py）
- `add_intercept(X)`：训练时扩展常数列。
- `split_coef(beta, fit_intercept)`：还原 `coef_` 与 `intercept_`。
- `clip_arr(arr, clip_range)`：通用裁剪。
- `converged(beta_old, beta_new, tol)`：收敛判定。

## 依赖边界
- 不使用 `jax.grad`/`vmap`/`hessian`。
- 仅用 `jax.numpy` 与基础线性代数求解器。

## 扩展指引
- **L1 / ElasticNet 支持**：当前架构基于二阶/最小二乘更新，天然支持 L2。若需支持 L1（不可导），建议新增基于 **Proximal Gradient** (FISTA) 或 **Coordinate Descent** 的 Solver。这些 Solver 依然可以复用 Formula 提供的梯度 (`z_resid` 相关) 和 Hessian 信息 (`W`)。
