# solvers 设计细化（IRLS / Newton-CG / SGD）

目标：提供显式迭代求解器，不依赖 autograd/vmap/hessian。
**特别约束**：不使用 `jnp.linalg.solve` 或矩阵分解 API，必须使用 Naive 的矩阵求逆 (`inv`) 后相乘的方式，以适配特定的后端（如 MPC）需求。

## 数学原理与推导

### 1. 变量定义
- **$y$**: 响应变量（Target），$y_i$ 为第 $i$ 个样本的观测值。
- **$X$**: 设计矩阵（Features），$x_i$ 为第 $i$ 个样本的特征向量（行向量）。
- **$\beta$**: 模型系数向量。
- **$\eta$**: 线性预测子 (Linear Predictor)，$\eta_i = x_i^T \beta$。
- **$\mu$**: 期望响应，$\mu_i = E[y_i]$。
- **$g(\cdot)$**: 链接函数 (Link Function)，$\eta_i = g(\mu_i)$，即 $\mu_i = g^{-1}(\eta_i)$。
- **$V(\mu)$**: 分布的方差函数 (Variance Function)，$Var(y_i) = \phi V(\mu_i)$，其中 $\phi$ 为离散度参数。

### 2. 指数族分布与对数似然
广义线性模型假设 $y$ 服从指数族分布，其概率密度函数可写为：
$$ f(y; \theta, \phi) = \exp\left( \frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi) \right) $$
其中 $\theta$ 是自然参数，与 $\mu$ 的关系为 $\mu = b'(\theta)$。
对数似然函数 $L$ (针对单个样本，忽略常数项) 为：
$$ L_i = \frac{y_i \theta_i - b(\theta_i)}{a(\phi)} $$

### 3. 梯度推导 (Score Function)
利用链式法则计算 $L_i$ 关于 $\beta_j$ 的导数：
$$ \frac{\partial L_i}{\partial \beta_j} = \frac{\partial L_i}{\partial \theta_i} \cdot \frac{\partial \theta_i}{\partial \mu_i} \cdot \frac{\partial \mu_i}{\partial \eta_i} \cdot \frac{\partial \eta_i}{\partial \beta_j} $$

分解各项：
1.  $\frac{\partial L_i}{\partial \theta_i} = \frac{y_i - b'(\theta_i)}{a(\phi)} = \frac{y_i - \mu_i}{a(\phi)}$
2.  $\frac{\partial \theta_i}{\partial \mu_i} = (\frac{\partial \mu_i}{\partial \theta_i})^{-1} = (b''(\theta_i))^{-1} = \frac{1}{V(\mu_i)}$
3.  $\frac{\partial \mu_i}{\partial \eta_i} = (g'(\mu_i))^{-1} = \frac{1}{g'(\mu_i)}$
4.  $\frac{\partial \eta_i}{\partial \beta_j} = x_{ij}$

合并得到（设 $a(\phi)=1$ 或归入系数）：
$$ \frac{\partial L_i}{\partial \beta} = (y_i - \mu_i) \cdot \frac{1}{V(\mu_i) g'(\mu_i)} \cdot x_i $$

为了方便计算，我们引入工作变量：
- **工作权重 (Working Weights)**: $W_i = \frac{1}{V(\mu_i) (g'(\mu_i))^2}$
- **工作残差 (Working Residuals)**: $z_{resid, i} = (y_i - \mu_i) g'(\mu_i)$

注意：
$$ W_i \cdot z_{resid, i} = \frac{1}{V (g')^2} \cdot (y-\mu) g' = \frac{y-\mu}{V g'} $$
这正是梯度公式中的中间项。因此，总梯度（Score 向量）为：
$$ \nabla_\beta L = \sum_i (W_i \cdot z_{resid, i}) x_i = X^T (W \odot z_{resid}) $$

### 4. Hessian 推导 (Fisher Information)
Fisher Information Matrix 是对数似然二阶导数的负期望：
$$ I = E\left[ - \frac{\partial^2 L}{\partial \beta \partial \beta^T} \right] $$
在 GLM 中（使用 Canonical Link 时观测 Hessian 等于期望 Hessian），其形式为：
$$ I = X^T W X $$
其中 $W$ 是对角矩阵，对角元为 $W_i$。

---

## IRLS (Iterative Reweighted Least Squares)

### 算法推导
IRLS 本质上是用于最大化似然函数的 **Fisher Scoring** 方法（即使用期望 Hessian 的 Newton-Raphson 法）。
Newton 更新规则：
$$ \beta_{new} = \beta_{old} + I^{-1} (\nabla_\beta L) $$
代入 $I = X^T W X$ 和 $\nabla_\beta L = X^T W z_{resid}$：
$$ \beta_{new} = \beta_{old} + (X^T W X)^{-1} X^T W z_{resid} $$
$$ (X^T W X) \beta_{new} = (X^T W X) \beta_{old} + X^T W z_{resid} $$
$$ (X^T W X) \beta_{new} = X^T W (X \beta_{old} + z_{resid}) $$

定义 **工作响应 (Working Response)** $z = X \beta_{old} + z_{resid} = \eta + z_{resid}$，则有：
$$ (X^T W X) \beta_{new} = X^T W z $$
这正是一个加权最小二乘问题（Weighted Least Squares）的正规方程。

### 更新规则 (Update Rule)
1.  计算 $W$ 和 $z_{resid}$。
2.  计算 $z = \eta + z_{resid}$。
3.  构建加权矩阵 $H = X^T W X$ 和加权向量 $score = X^T W z$。
4.  **L2 正则化处理**:
    - 若存在 L2 正则 (强度 $\lambda$)，则更新 $H \leftarrow H + \lambda I$。
    - 注意：Intercept 对应的对角线元素通常不加正则。
5.  Naive Update: $\beta_{new} = (H + \epsilon I)^{-1} score$。
    - 这里 $\epsilon$ 是数值稳定性抖动 (Jitter)。
    - 注意：Score 向量 $X^T W z$ **不需要** 减去 $\lambda \beta$，因为梯度中的 $-\lambda \beta$ 已在 Newton 迭代推导中抵消（详见 `api_design.md`）。

---

## SGD (Stochastic Gradient Descent)

### 损失函数与梯度
SGD 通常最小化负对数似然 (NLL) 或 Deviance。这里我们表述为 **梯度上升 (Gradient Ascent)** 最大化对数似然 $L$。

目标函数：$J(\beta) = L(\beta) - \frac{\lambda}{2} \|\beta\|_2^2$ (含 L2 正则)。

根据前述推导，单个样本的梯度为：
$$ \nabla_\beta L_i = x_i \cdot \frac{y_i - \mu_i}{V(\mu_i) g'(\mu_i)} = x_i \cdot (W_i \cdot z_{resid, i}) $$

### 更新规则 (Update Rule)
对于一个 Batch $B$：
1.  计算 Batch 内的 $W_B$ 和 $z_{resid, B}$。
2.  计算 Batch 梯度：
    $$ g_B = \sum_{i \in B} x_i (W_i \cdot z_{resid, i}) = X_B^T (W_B \odot z_{resid, B}) $$
3.  添加正则项梯度（Intercept 除外）：
    $$ g_{total} = g_B - \lambda \cdot \beta $$
4.  参数更新（Gradient Ascent）：
    $$ \beta_{t+1} = \beta_t + \eta \cdot g_{total} $$
    其中 $\eta$ 是学习率。

---

## 统一接口定义
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
        tol: float = 1e-4,
        learning_rate: float = 1e-2,  # SGD specific
        batch_size: int = 128,        # SGD specific
        clip_eta=None,
        clip_mu=None,
    ) -> tuple[beta, dispersion, history]:
        ...
```

## 数值要点
- **Naive Matrix Inversion**:
    - 所有涉及线性方程组求解的地方（主要是 IRLS 中的 Update），**必须** 使用 `inv(H) @ z` 的形式。
    - **禁止** 使用 `solve`, `cholesky`, `qr`。
    - 必须在求逆前对矩阵对角线添加 `epsilon` 保证可逆。

