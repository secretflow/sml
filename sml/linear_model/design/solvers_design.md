# solvers 设计细化（IRLS / Newton）

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

## IRLS
核心步骤：
1. 初始化 `beta = 0`，可用 `family.distribution.starting_mu(y)` 推断初值。
2. 迭代直到收敛或 `max_iter`：
   - `eta = X @ beta + offset`
   - 调 `formula.compute_components(...)` 得到 `W, z_resid, mu, eta`
   - `z = eta + z_resid`
   - 构造带权矩阵：`Xw = sqrt(W) * X`，`zw = sqrt(W) * z`
   - 正则：对除 intercept 以外的列加 `l2` 对角抬升
   - 更新：`beta_new = solve( Xw^T Xw + l2I , Xw^T zw )`（用 `jnp.linalg.solve` 或 Cholesky）
   - 收敛：`max(|beta_new - beta|) / (max(|beta|)+eps) < tol`
3. dispersion 估计：
   - 对指数族常见做法：`phi = deviance / dof_resid`（可选，按分布决定是否需要）。

数值要点：
- 对 `W` 设下界 `w_min`，避免奇异矩阵。
- 对 `eta`、`mu` 使用传入的 clip 防溢出。
- 拆分 intercept 正则：在对角抬升时对 intercept 位置置 0。

## Newton-Raphson（显式一二阶）
- `eta = X @ beta + offset`
- `mu = link.inverse(eta)`
- `score = X^T * ((y - mu) * link.link_deriv(mu) / V(mu))`
- `H = X^T * diag(link.link_deriv(mu)**2 / V(mu)) * X`
- 正则同 IRLS，对非截距项加 `l2`。
- 更新：`beta_new = beta + solve(H + l2I, score)`
- 收敛准则同 IRLS。

数值要点：
- 对 `H` 可加阻尼（damping）以提升稳定性。
- 允许最大步长限制，防止过冲。

## 工具函数（solvers/utils.py）
- `add_intercept(X)`：训练时扩展常数列。
- `split_coef(beta, fit_intercept)`：还原 `coef_` 与 `intercept_`。
- `clip_arr(arr, clip_range)`：通用裁剪。
- `converged(beta_old, beta_new, tol)`：收敛判定。

## 依赖边界
- 不使用 `jax.grad`/`vmap`/`hessian`。
- 仅用 `jax.numpy` 与基础线性代数求解器。

## 扩展指引
- 若新增 solver，保持 `solve` 签名一致，复用 utils。