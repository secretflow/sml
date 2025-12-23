# MPC 下 GLM 标签缩放 (Scaling) 技术分析

**版本**: 1.0
**背景**: SML (Secure Machine Learning) 库实现参考
**目标**: 分析在 MPC 定点数环境下，对目标变量 $y$ 进行线性缩放（如 $y' = y / K$）的可行性、理论影响及工程注意事项。

---

## 1. 核心结论

从理论上讲，对 $y$ 进行线性缩放（$y' = y / K$）**不会破坏 GLM 的核心预测能力**，也不会引入预测偏差（Bias），前提是关注点估计（Point Estimation，即预测均值 $\mu$）。

*   **优点**: 防止 MPC 定点数计算中的溢出 (Overflow)，使数据落入数值计算的“甜点区”。
*   **代价**: 会改变离散参数（Dispersion Parameter $\phi$）的数值解释，并可能影响正则化参数 $\lambda$ 的最佳取值范围。

---

## 2. 理论分析：指数族分布的标度不变性

GLM 假设 $y$ 服从指数族分布，方差满足 $\text{Var}(y) = \phi V(\mu)$。令 $y' = c \cdot y$ （其中 $c = 1/K$），分析各分布的性质变化。

### 2.1 Gamma 分布 (标度不变性)
*   **原性质**: $\text{Var}(y) = \phi \mu^2$
*   **缩放后**:
    $$ \text{Var}(y') = c^2 \text{Var}(y) = c^2 \phi \mu^2 = \phi (c\mu)^2 = \phi (\mu')^2 $$
*   **结论**: $y'$ 依然完美服从 Gamma 分布，且离散参数 $\phi$ **保持不变**。这是最理想的情况。

### 2.2 Tweedie 分布 ($1 < p < 2$)
*   **原性质**: $\text{Var}(y) = \phi \mu^p$
*   **缩放后**:
    $$ \text{Var}(y') = c^2 \phi \mu^p = c^2 \phi \left(\frac{\mu'}{c}\right)^p = (c^{2-p}\phi) (\mu')^p $$
*   **结论**: $y'$ 依然服从 Tweedie 分布形式，离散参数变为 $\phi' = c^{2-p}\phi$。这对 $\boldsymbol{\beta}$ 的点估计无影响。

### 2.3 正态分布 (Gaussian)
*   **原性质**: $\text{Var}(y) = \sigma^2$
*   **缩放后**: $\text{Var}(y') = c^2 \sigma^2$
*   **结论**: 依然是正态分布，方差缩小 $c^2$ 倍。

### 2.4 泊松分布 (Poisson) —— 需特别注意
*   **原性质**: $\text{Var}(y) = \mu$ ($\phi=1$)，定义域为非负整数。
*   **缩放后**: $y'$ 变为小数，且 $\text{Var}(y') = c^2 \mu = c \mu'$。
*   **结论**: 这不再是标准泊松分布，而是 **拟泊松 (Quasi-Poisson)** 分布，其离散参数 $\phi = c$。
*   **影响**: GLM 求解本质上是最大化拟似然 (Quasi-likelihood)。只要方差结构 $V(\mu) \propto \mu$ 保持不变，点估计 $\hat{\boldsymbol{\beta}}$ 依然是无偏的。可以直接使用 Poisson Loss 进行训练。

---

## 3. 对模型系数 $\boldsymbol{\beta}$ 的影响

假设原模型 $\eta = \mathbf{X}\boldsymbol{\beta}, \mu = g^{-1}(\eta)$，缩放后模型 $\eta' = \mathbf{X}\boldsymbol{\beta}', \mu' = c\mu$。

### 3.1 Log Link (常用：Tweedie, Gamma, Poisson)
$$ \log(\mu') = \log(c\mu) = \log(c) + \log(\mu) = \underbrace{\log(c) + \beta_0}_{\beta'_0} + \sum_{j>0} \beta_j x_j $$
*   **结论**: **仅截距项 (Intercept) 发生平移**，特征权重 (Slope) $\beta_j (j>0)$ **完全不变**。
*   **优势**: 非常适合 MPC，因为特征系数的数值范围稳定，不会因 $y$ 的量级变化而剧烈波动。

### 3.2 Identity Link (Linear Regression)
$$ \mu' = c\mu \implies \mathbf{X}\boldsymbol{\beta}' = c(\mathbf{X}\boldsymbol{\beta}) $$
*   **结论**: 所有系数（包括截距）整体缩小 $c$ 倍。

---

## 4. MPC 工程风险与解决方案

### 4.1 正则化系数失配 (Regularization Mismatch)
目标函数通常为 $\mathcal{L}(\boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|^2$。
*   **问题**: $y$ 缩小后，似然函数部分的梯度会缩小，但正则化项梯度 $-\lambda \boldsymbol{\beta}$ 不变（对于 Log Link，$\beta$ 大小不变）。这会导致正则化相对权重变大，引发**欠拟合 (Underfitting)**。
*   **修正**: 需同步调整 $\lambda$。
    *   **Linear Regression**: $\lambda_{new} \approx c^2 \lambda_{old}$
    *   **Log Link**: 梯度变小了但 $\beta$ 没变，建议调小 $\lambda$ (具体比例需根据 Loss 函数形式推导或实验确定)。

### 4.2 精度损失 (Precision Loss)
*   **问题**: 若 $y$ 极小 (长尾分布)，除以 $y_{max}$ 后可能导致下溢 (Underflow) 为 0。
*   **建议**: 不要无脑除以 $y_{max}$。可考虑除以 $p95$ 或 $p99$ 分位数，确保大部分 $y'$ 落在 $[0.01, 10]$ 的定点数甜点区。

### 4.3 恢复误差 (Recovery Error)
*   **问题**: 最终预测 $\hat{y} = \hat{y}' \times K$。若在 MPC 密态下进行乘法，可能再次溢出。
*   **建议**: **在明文下恢复**。将 $\hat{y}'$ Reveal 给用户，由用户在本地乘回 $K$。

---

## 5. 实施建议速查

1.  **预处理**: $y_{train} = y_{raw} / \text{scale}$。
2.  **分布配置**:
    *   Gamma/Tweedie/Gaussian: 直接训练。
    *   Poisson: 允许输入非整数，注明为 Quasi-Poisson 回归。
3.  **Link 选择**: 首选 **Log Link** (保持特征系数稳定性)。
4.  **超参**: 提示用户若开启 Scaling，需适当减小 `l2_norm`。
5.  **后处理**: 输出 $\hat{y}'$ 及 `scale`，用户本地执行 $\hat{y} = \hat{y}' \times \text{scale}$。
