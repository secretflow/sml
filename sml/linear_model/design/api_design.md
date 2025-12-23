# API Design: Scale (Dispersion) Handling in GLM IRLS

本文档深入探讨广义线性模型 (GLM) 中尺度参数 (Scale/Dispersion Parameter, $\phi$) 在迭代重加权最小二乘法 (IRLS) 中的作用、数学推导、数值影响以及最终的 API 设计决策。

## 1. IRLS 基础原理

迭代重加权最小二乘法 (IRLS, Iteratively Reweighted Least Squares) 是求解 GLM 最大似然估计 (MLE) 的标准数值方法。它本质上是应用于对数似然函数的 **Newton-Raphson** (或 Fisher Scoring) 优化算法。

GLM 假设响应变量 $y$ 服从指数族分布，其对数似然函数 (Log-Likelihood) 与线性预测子 $\eta = X\beta$ 非线性相关。IRLS 通过将非线性优化问题转化为一系列加权最小二乘 (WLS) 问题来逼近最优解。

## 2. 权重推导 (Derivation of Weights)

### 2.1 基础定义
- **方差函数**: $Var(y_i) = \phi V(\mu_i)$，其中 $\phi$ 是尺度参数 (Dispersion)，$V(\cdot)$ 是由分布决定的方差函数。
- **链接函数**: $\eta_i = g(\mu_i)$，其导数为 $g'(\mu_i) = \frac{d\eta}{d\mu}$。

### 2.2 似然梯度 (Score)
对数似然 $L$ 关于 $\beta$ 的梯度 (Score Vector) 为：
$$ \nabla_\beta L = \sum_i \frac{y_i - \mu_i}{Var(y_i)} \frac{\partial \mu_i}{\partial \eta_i} x_i $$
代入方差和链接函数：
$$ \nabla_\beta L = \sum_i \frac{y_i - \mu_i}{\phi V(\mu_i)} \frac{1}{g'(\mu_i)} x_i = \frac{1}{\phi} X^T \mathbf{W}_{can} \mathbf{z}_{resid} $$
其中：
- $\mathbf{W}_{can} = \text{diag}\left( \frac{1}{V(\mu) (g'(\mu))^2} \right)$ 是**典范权重 (Canonical Weights)**（不含 $\phi$）。
- $\mathbf{z}_{resid} = (y - \mu) g'(\mu)$ 是工作残差。

### 2.3 Fisher 信息矩阵 (Hessian)
期望 Hessian 矩阵 (Fisher Information Matrix) 为：
$$ \mathbf{H} = E\left[ - \frac{\partial^2 L}{\partial \beta \partial \beta^T} \right] = X^T \mathbf{W}_{total} X $$
其中总权重 $\mathbf{W}_{total}$ 包含尺度参数：
$$ W_{total, i} = \frac{1}{Var(y_i) (g'(\mu_i))^2} = \frac{1}{\phi V(\mu_i) (g'(\mu_i))^2} = \frac{1}{\phi} W_{can, i} $$
因此：
$$ \mathbf{H} = \frac{1}{\phi} X^T \mathbf{W}_{can} X $$

## 3. 标准 IRLS 更新公式

Newton-Raphson 更新规则为：
$$ \beta^{(t+1)} = \beta^{(t)} + \mathbf{H}^{-1} (\nabla_\beta L) $$

代入上述推导：
$$ \beta^{(t+1)} = \beta^{(t)} + \left( \frac{1}{\phi} X^T \mathbf{W}_{can} X \right)^{-1} \left( \frac{1}{\phi} X^T \mathbf{W}_{can} \mathbf{z}_{resid} \right) $$

令工作响应 $\mathbf{z} = \eta + \mathbf{z}_{resid} = X\beta^{(t)} + \mathbf{z}_{resid}$，上述公式可重写为加权最小二乘形式：
$$ \beta^{(t+1)} = \left( \frac{1}{\phi} X^T \mathbf{W}_{can} X \right)^{-1} \left( \frac{1}{\phi} X^T \mathbf{W}_{can} \mathbf{z} \right) $$

## 4. Scale 抵消现象 (Scale Cancellation)

### 4.1 数学证明
观察上述更新公式中的项：
$$ \left( \frac{1}{\phi} X^T \mathbf{W}_{can} X \right)^{-1} = \left( \frac{1}{\phi} \mathbf{A} \right)^{-1} = \phi \mathbf{A}^{-1} $$
其中 $\mathbf{A} = X^T \mathbf{W}_{can} X$。

代入更新公式：
$$ \text{Update Term} = \left( \phi (X^T \mathbf{W}_{can} X)^{-1} \right) \left( \frac{1}{\phi} X^T \mathbf{W}_{can} \mathbf{z}_{resid} \right) $$
由于 $\phi$ 是标量且非零，$\phi \cdot \frac{1}{\phi} = 1$。
$$ \text{Update Term} = (X^T \mathbf{W}_{can} X)^{-1} (X^T \mathbf{W}_{can} \mathbf{z}_{resid}) $$

### 4.2 结论
**在不含正则化的标准 IRLS 中，尺度参数 $\phi$ 在更新公式中完全抵消。**
这意味着：
1.  求解回归系数 $\beta$ 不需要预先知道或估计 $\phi$。
2.  我们可以假设 $\phi=1$ 进行计算，得到的结果与真实 $\phi$ 值无关。
3.  $\phi$ 仅在模型训练完成后，用于计算标准误 (Standard Errors)、置信区间或 p 值时才需要估计。

## 5. 正则化影响 (Effect of Regularization)

当引入 L2 正则化 (Ridge) 时，目标函数变为：
$$ Q(\beta) = L(\beta) - \frac{\lambda}{2} \|\beta\|_2^2 $$
这里 $\lambda$ 是用户指定的正则化强度。

梯度：
$$ \nabla Q = \frac{1}{\phi} X^T \mathbf{W}_{can} \mathbf{z}_{resid} - \lambda \beta $$
Hessian：
$$ \mathbf{H}_Q = -\frac{1}{\phi} X^T \mathbf{W}_{can} X - \lambda I $$

更新公式：
$$ \Delta \beta = -\mathbf{H}_Q^{-1} \nabla Q = \left( \frac{1}{\phi} X^T \mathbf{W}_{can} X + \lambda I \right)^{-1} \left( \frac{1}{\phi} X^T \mathbf{W}_{can} \mathbf{z}_{resid} - \lambda \beta \right) $$

为了简化，分子分母同乘 $\phi$：
$$ \Delta \beta = \left( X^T \mathbf{W}_{can} X + \phi \lambda I \right)^{-1} \left( X^T \mathbf{W}_{can} \mathbf{z}_{resid} - \phi \lambda \beta^{(t)} \right) $$

**最终更新公式**:
$$ \beta^{(t+1)} = \beta^{(t)} + \Delta \beta $$
$$ \beta^{(t+1)} = \beta^{(t)} + (X^T \mathbf{W}_{can} X + \lambda' I)^{-1} (X^T \mathbf{W}_{can} \mathbf{z}_{resid} - \lambda' \beta^{(t)}) $$
其中 $\lambda' = \phi \lambda$。

为了推导闭式更新，我们将 $\beta^{(t)}$ 乘上单位矩阵 $(X^T \mathbf{W}_{can} X + \lambda' I)^{-1} (X^T \mathbf{W}_{can} X + \lambda' I)$：
$$ \beta^{(t+1)} = (X^T \mathbf{W}_{can} X + \lambda' I)^{-1} \left[ (X^T \mathbf{W}_{can} X + \lambda' I) \beta^{(t)} + X^T \mathbf{W}_{can} \mathbf{z}_{resid} - \lambda' \beta^{(t)} \right] $$
展开方括号内各项：
$$ = X^T \mathbf{W}_{can} X \beta^{(t)} + \lambda' \beta^{(t)} + X^T \mathbf{W}_{can} \mathbf{z}_{resid} - \lambda' \beta^{(t)} $$
注意 $\lambda' \beta^{(t)}$ 正负抵消：
$$ = X^T \mathbf{W}_{can} (X \beta^{(t)} + \mathbf{z}_{resid}) $$
利用工作响应定义的 $\mathbf{z} = X \beta^{(t)} + \mathbf{z}_{resid}$：
$$ \beta^{(t+1)} = (X^T \mathbf{W}_{can} X + \lambda' I)^{-1} X^T \mathbf{W}_{can} \mathbf{z} $$

**结论**: 正确的 L2 Regularized IRLS 更新公式中，右侧 Score 向量 **不包含** $-\lambda \beta$ 项。这是因为 Newton 步中的梯度项 $-\lambda \beta$ 与 Hessian 修正带来的 $+\lambda \beta$ 相互抵消。

### 5.1 分析
此时，**$\phi$ 不再抵消**。它变成了正则化项的缩放因子。
实际上，我们通常将 $\lambda' = \phi \lambda$ 视为一个整体的超参数（effective regularization strength）。在实际应用（如 `sklearn` 或 SML）中，用户通过交叉验证选择 `l2` 参数，这个 `l2` 实际上隐含了对 $\phi$ 的适应。因此，**在算法实现层面，我们依然可以固定设 $\phi=1$，并将用户传入的 `l2` 视为 $\lambda'$。**

## 6. GLM 场景案例分析

### 6.1 Gamma 分布 + Log Link
- **Variance**: $V(\mu) = \mu^2$
- **Link**: $g(\mu) = \log(\mu) \Rightarrow g'(\mu) = 1/\mu$
- **Weight (Unscaled)**: $W_{can} = \frac{1}{\mu^2 (1/\mu)^2} = 1$
- **Scale $\phi$**: Gamma 分布的 $\phi$ (即 $1/\nu$) 通常未知。
- **现象**: 即使数据非常离散 ($\phi$ 很大)，IRLS 求解 $\beta$ 的路径不受影响。权重恒为 1 (常数权重)，IRLS 退化为迭代求解线性方程组。

### 6.2 Tweedie 分布 (p=1.5) + Log Link
- **Variance**: $V(\mu) = \mu^p = \mu^{1.5}$
- **Link**: $g'(\mu) = 1/\mu$
- **Weight**: $W_{can} = \frac{1}{\mu^{1.5} (1/\mu)^2} = \mu^{0.5} = \sqrt{\mu}$
- **Scale $\phi$**: 同样在更新中抵消。

## 7. 数值稳定性与 Scale

虽然理论上 $\phi$ 抵消，但在数值计算中需要注意：

1.  **矩阵求逆**: 在 SecretFlow 等 MPC 场景下，使用 Naive Matrix Inversion (`inv(H)`)。如果 $\phi$ 极小（例如 $10^{-5}$）且未被约去，会导致 Hessian 元素极大，求逆可能溢出；反之 $\phi$ 极大导致 Hessian 极小，求逆不稳定。
2.  **SML/JAX 策略**: 通过假设 $\phi=1$ 计算 $W_{can}$，我们保证了 $H = X^T W_{can} X$ 的量级主要由 $X$ 和 $\mu$ 决定，避免了人为引入 $\phi$ 带来的数值波动。
3.  **Jitter**: 为了保证 $H$ 可逆，我们总是添加 $\epsilon I$ (`H + eps * I`)。这里的 $\epsilon$ 实际上也扮演了微小正则化的角色，其相对强度会受到隐式 $\phi$ 的影响，但在双精度/单精度下通常可忽略。

## 8. 代码实现点 (SML/JAX-GLM)

### 8.1 权重计算 (`formula/generic.py`)
```python
# SML 计算的是不含 phi 的 Canonical Weight
# w = 1.0 / (v_mu * (g_prime**2) + eps)
w = 1.0 / (v_mu * (g_prime**2) + eps)
```
这里隐含了 $\phi=1$。

### 8.2 IRLS 更新 (`solvers/irls.py`)
```python
# H = X^T * W * X (即 X^T * W_can * X)
H = Xw.T @ Xw
# Score = X^T * W * z (即 X^T * W_can * z)
score = Xw.T @ zw

# 正则化直接加在 H 上，相当于公式中的 lambda'
if l2 > 0:
    H = H.at[diag_indices].add(l2)

# 更新
beta_new = invert_matrix(H) @ score
```
代码实现完全符合上述“隐含 $\phi=1$”的设计，将 `l2` 视为有效正则化强度。

## 9. API 设计总结与原因

结合 SML (JAX) 和 SecretFlow (SS-GLM) 的背景，我们的 API 设计决策如下：

### 9.1 设计决策
1.  **Solver 接口不接受 `dispersion` / `scale` 参数**。
2.  **Formula 计算 `W` 时默认 $\phi=1$**。
3.  **正则化参数 `l2` 定义为绝对惩罚项**，即对应公式中的 $\lambda \phi$。
4.  **Dispersion 仅作为结果输出** (`model.dispersion_`)，在 `fit` 结束后通过 Pearson $\chi^2$ 或 Deviance 估计，用于后续统计推断（如有），但不参与 $\beta$ 的迭代优化。

### 9.2 原因总结
1.  **数学简洁性 (Cancellation)**: 如前所述，$\phi$ 在无正则化时完全抵消，引入它只会增加计算复杂度和数值风险。
2.  **超参统一性**: 在正则化场景下，用户通常通过验证集搜索最佳超参。区分 `lambda` 和 `phi * lambda` 对用户来说没有意义，直接暴露单一的 `l2` 参数更符合机器学习库（如 sklearn）的习惯。
3.  **SecretFlow 适配**:
    - 在 MPC 环境中，除法和求逆是昂贵操作。避免在迭代中重复计算或Reveal $\phi$ 可以提升性能。
    - SS-GLM 目前的实现中，虽然部分公式显式写出了 `scale`，但如果在输入时未提供且无法准确估计，通常也是设为 1.0 处理。SML 的设计通过公式层面的简化，使得未来移植到 MPC 后端时更加高效且数值稳定。
4.  **解耦**: 将“模型拟合 ($\\beta$)”与“统计推断 ($\\phi$, p-value)”解耦。核心 Solver 只关注 $\\beta$ 的收敛。

## 10. 关于 $a(\phi)$ 函数 (The a(.) function)

在 GLM 的指数族分布定义中，$a(\phi)$ 是一个用于缩放对数似然的函数。

### 10.1 数学定义
$$ f(y; \theta, \phi) = \exp\left( \frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi) \right) $$
通常形式为：
$$ a(\phi) = \frac{\phi}{w} $$
其中：
- $\phi$ 是离散度参数 (Dispersion/Scale)。
- $w$ 是样本权重 (Prior Weights)，默认为 1。

对于常见分布：
- **Gaussian, Inverse Gaussian, Gamma, Tweedie**: $a(\phi) = \phi / w$。
- **Poisson, Binomial, Negative Binomial (fixed)**: $\phi=1$ (固定)，故 $a(\phi) = 1/w$。

### 10.2 在 SML 中的对应
在我们的实现 (`Formula` 和 `Distribution`) 中：
1.  **样本权重 $w$**: 直接乘在 `W` (Working Weights) 上。
    - $W_{core} = \frac{1}{V(\mu) (g'(\mu))^2}$
    - $W_{final} = W_{core} \cdot w$
2.  **离散度 $\phi$**: 如前文所述，在 IRLS 更新中被抵消 (implied $\phi=1$)。

因此，我们的实现隐含了 $a(\phi) = 1/w$ (在计算 $W$ 时) 或者说 $\phi$ 因子被提取到方程两边消去。