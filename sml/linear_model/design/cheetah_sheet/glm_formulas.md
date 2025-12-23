# GLM 算法核心公式速查表 (GLM Formula Cheat Sheet)

**版本**: 1.0
**背景**: SML (Secure Machine Learning) 库实现参考
**目标**: 提供基于矩阵运算的广义线性模型 (GLM) 核心公式推导，包含 L2 正则化、样本权重及离散参数处理，涵盖 IRLS 和 SGD 两种求解器。

---

## 1. 符号体系 (Notation System)

为了保证代码与文档的一致性，采用以下矩阵符号定义。

### 1.1 维度定义
*   $N$: 样本数量 (Samples)。
*   $p$: 特征数量 (Features)，包含截距项。

### 1.2 数据与参数

| 符号 | 维度 | 含义 | 备注 |
| :--- | :--- | :--- | :--- |
| $\mathbf{X}$ | $N \times p$ | 设计矩阵 (Design Matrix) | 第 $i$ 行记为 $\mathbf{x}_i^T$ |
| $\mathbf{y}$ | $N \times 1$ | 响应变量 (Response Vector) | $\mathbf{y} = [y_1, \dots, y_N]^T$ |
| $\boldsymbol{\beta}$ | $p \times 1$ | 回归系数 (Coefficients) | 待优化参数 |
| $\mathbf{w}$ | $N \times 1$ | 样本权重 (Sample Weights) | 用户输入权重 (默认为 $\mathbf{1}$) |
| $\boldsymbol{\eta}$ | $N \times 1$ | 线性预测子 (Linear Predictor) | $\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\beta} + \text{offset}$ |
| $\boldsymbol{\mu}$ | $N \times 1$ | 期望均值 (Expected Mean) | $\mu_i = E[y_i]$ |

### 1.3 函数与统计量

| 符号 | 类型 | 含义 | 数学定义 |
| :--- | :--- | :--- | :--- |
| $g(\cdot)$ | Element-wise | 连接函数 (Link Function) | $\eta_i = g(\mu_i)$ |
| $g'(\cdot)$ | Element-wise | 连接函数一阶导数 | $g'(\mu) = \frac{d\eta}{d\mu}$ |
| $V(\cdot)$ | Element-wise | 方差函数 (Variance Function) | $\text{Var}(y_i) = \frac{\phi}{w_i} V(\mu_i)$ |
| $\phi$ | 标量 | 离散参数 (Dispersion Parameter) | 控制分布的方差缩放 |
| $\lambda$ | 标量 | L2 正则化系数 | 惩罚项强度 |
| $\alpha$ | 标量 | 学习率 (Learning Rate) | 仅用于 SGD |

---

## 2. 优化目标函数 (Objective Function)

GLM 的目标是最小化 **带 L2 正则化的负对数似然 (Regularized Negative Log-Likelihood, NLL)**。
对于指数族分布 (EDF)，目标函数 $J(\boldsymbol{\beta})$ 定义为：

$$ J(\boldsymbol{\beta}) = \underbrace{- \sum_{i=1}^N \frac{w_i}{\phi} \left( y_i \theta_i - b(\theta_i) \right)}_{\text{Negative Log-Likelihood}} + \underbrace{\frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2}_{\text{L2 Penalty}}
$$ 

*   $\theta_i$: 典则参数 (Canonical parameter)，满足 $b'(\theta_i) = \mu_i$ 且 $b''(\theta_i) = V(\mu_i)$。
*   **关于 $\phi$ 的重要说明**: 离散参数 $\phi$ 出现在似然项的分母中。在优化推导中，这会影响正则化项的相对权重。下文公式已做显式处理以确保数值稳定性。

---

## 3. 通用计算框架 (General Framework)

### 3.1 离散参数估计 ($\phi$)
在 $\boldsymbol{\beta}$ 优化过程中，$\phi$ 通常视为常数。每轮优化结束后，使用 **皮尔逊卡方统计量 (Pearson Chi-squared statistic)** 进行更新：

$$ \hat{\phi} = \frac{1}{N - p} \sum_{i=1}^N \frac{w_i (y_i - \mu_i)^2}{V(\mu_i)}
$$ 

*   对于 Poisson/Binomial 分布，理论上 $\phi=1$，但计算该值有助于检测过离散 (Over-dispersion) 现象。

---

### 3.2 算法一：IRLS (迭代加权最小二乘)

IRLS 等价于牛顿法 (Newton-Raphson)。利用 Hessian 矩阵构建二阶更新步。

#### 步骤 1: 构建工作矩阵 (Working Matrices)

计算当前 $\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\beta}$ 和 $\boldsymbol{\mu} = g^{-1}(\boldsymbol{\eta})$。

1.  **工作权重矩阵 ($\mathbf{W}$)**  
    $N \times N$ 对角矩阵：
    $$ \mathbf{W} = \mathrm{diag}\left( \frac{w_i}{V(\mu_i) (g'(\mu_i))^2} \right)
    $$ 

2.  **工作响应向量 ($\mathbf{z}$)**  
    $N \times 1$ 向量 (响应变量的一阶泰勒展开)：
    $$ z_i = \eta_i + (y_i - \mu_i) g'(\mu_i)
    $$ 
    *向量形式*: $\mathbf{z} = \boldsymbol{\eta} + \mathrm{diag}(g'(\boldsymbol{\mu})) (\mathbf{y} - \boldsymbol{\mu})$

#### 步骤 2: 参数更新 (求解线性方程组)

求解 $\boldsymbol{\beta}^{(t+1)}$ 本质上是求解带 L2 正则化的加权最小二乘问题。
标准的牛顿更新步 $\boldsymbol{\beta}_{new} = \boldsymbol{\beta}_{old} - \mathbf{H}^{-1}\nabla J$ 经化简后变为：

$$ \left( \mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \hat{\phi} \mathbf{I} \right) \boldsymbol{\beta}^{(t+1)} = \mathbf{X}^T \mathbf{W} \mathbf{z}
$$ 

**实现细节**:
*   **左侧 (LHS)**: $\mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \hat{\phi} \mathbf{I}$。注意正则化项系数变为 $\lambda \hat{\phi}$。这是因为原始 Hessian 包含 $1/\phi$ 因子，为了消除分母，方程两边同乘了 $\phi$。
*   **右侧 (RHS)**: $\mathbf{X}^T \mathbf{W} \mathbf{z}$。**切记**此处不要减去 $\lambda \boldsymbol{\beta}$。L2 梯度项已隐式包含在左侧的 $\lambda \hat{\phi} \mathbf{I}$ 中（直接求解新参数 $\boldsymbol{\beta}^{(t+1)}$）。
*   **求解器**: 使用 Cholesky 分解 (若 LHS 正定) 或 QR 分解。

---

### 3.3 算法二：SGD (随机梯度下降)

当 $N$ 过大无法构建 $\mathbf{X}^T \mathbf{W} \mathbf{X}$ 时使用 SGD。

#### 梯度计算
目标函数 $J(\boldsymbol{\beta})$ 的梯度为：

$$ \nabla J(\boldsymbol{\beta}) = -\frac{1}{\phi} \mathbf{X}^T \left( \mathbf{w} \odot \frac{\mathbf{y} - \boldsymbol{\mu}}{V(\boldsymbol{\mu}) \odot g'(\boldsymbol{\mu})} \right) + \lambda \boldsymbol{\beta}
$$ 

定义 **辅助向量 (Auxiliary Vector) $\mathbf{c}$**:
$$ c_i = \frac{w_i}{V(\mu_i) g'(\mu_i)}
$$ 

梯度简写为：
$$ \nabla J(\boldsymbol{\beta}) = -\frac{1}{\phi} \mathbf{X}^T (\mathbf{c} \odot (\mathbf{y} - \boldsymbol{\mu})) + \lambda \boldsymbol{\beta}
$$ 

#### 更新公式 (Weight Decay)
设学习率为 $\alpha$：

$$ \boldsymbol{\beta}^{(t+1)} = \underbrace{(1 - \alpha \lambda) \boldsymbol{\beta}^{(t)}}_{\text{Weight Decay}} + \underbrace{\frac{\alpha}{\hat{\phi}} \mathbf{X}_{batch}^T \left( \mathbf{c}_{batch} \odot (\mathbf{y}_{batch} - \boldsymbol{\mu}_{batch}) \right)}_{\text{Data Update}}
$$ 

---

## 4. 常见分布特化公式 (Specialized Formula Cheatsheet)

为了数值稳定性和计算效率，请直接使用以下化简后的公式构建 $\mathbf{W}$ 和 $\mathbf{z}$，**严禁**使用通用的导数链式计算。

### 4.1 Gaussian + Identity (线性回归)
*   **定义**: $V(\mu)=1, \quad \eta=\mu, \quad g'(\mu)=1$.
*   **IRLS 组件**:
    *   $\\mathbf{W} = \mathrm{diag}(\mathbf{w})$
    *   $\\mathbf{z} = \mathbf{y}$
*   **更新方程**:
    $$ (\mathbf{X}^T \mathrm{diag}(\mathbf{w}) \mathbf{X} + \lambda \hat{\sigma}^2 \mathbf{I}) \boldsymbol{\beta} = \mathbf{X}^T (\mathbf{w} \odot \mathbf{y})
    $$ 

### 4.2 Poisson + Log (典则连接)
*   **定义**: $V(\mu)=\mu, \quad \mu = e^\eta, \quad g'(\mu) = 1/\mu$.
*   **IRLS 组件**:
    *   $W_{ii} = w_i \mu_i$  
        *(推导: $w / (\mu \cdot (1/\mu)^2) = w\mu$)*
    *   $z_i = \eta_i + \frac{y_i}{\mu_i} - 1$
*   **SGD 辅助向量**: $c_i = w_i$ (梯度简化为 $\mathbf{X}^T(\mathbf{w} \odot (\mathbf{y}-\boldsymbol{\mu}))$).

### 4.3 Bernoulli + Logit (逻辑回归)
*   **定义**: $V(\mu)=\mu(1-\mu), \quad \mu = \sigma(\eta), \quad g'(\mu) = \frac{1}{\mu(1-\mu)}$.
*   **IRLS 组件**:
    *   $W_{ii} = w_i \mu_i (1 - \mu_i)$
    *   $z_i = \eta_i + \frac{y_i - \mu_i}{\mu_i(1-\mu_i)}$
*   **SGD 辅助向量**: $c_i = w_i$ (梯度简化为 $\mathbf{X}^T(\mathbf{w} \odot (\mathbf{y}-\boldsymbol{\mu}))$).

### 4.4 Gamma + Log (业务常用)
*   **定义**: $V(\mu)=\mu^2, \quad \mu = e^\eta, \quad g'(\mu) = 1/\mu$.
*   **IRLS 组件**:
    *   $W_{ii} = w_i$ (**常数!**) 
        *(推导: $w / (\mu^2 \cdot (1/\mu)^2) = w$)*
        *优化点*: Hessian 核心 $\mathbf{X}^T \mathbf{W} \mathbf{X}$ 是固定的（若忽略 $\lambda \phi$ 变化），可预计算或缓存分解结果。
    *   $z_i = \eta_i + \frac{y_i}{\mu_i} - 1$
*   **SGD 辅助向量**: $c_i = w_i / \mu_i$.

### 4.5 Tweedie ($p \in (1, 2)$) + Log
*   **定义**: $V(\mu)=\mu^p, \quad \mu = e^\eta, \quad g'(\mu) = 1/\mu$.
*   **IRLS 组件**:
    *   $W_{ii} = w_i \cdot \mu_i^{2-p}$
        *技巧*: 计算为 $w_i \cdot \exp(\eta_i(2-p))$ 以避免中间值溢出。
    *   $z_i = \eta_i + y_i e^{-\eta_i} - 1$
*   **SGD 辅助向量**: $c_i = w_i \cdot \mu_i^{1-p}$.

### 4.6 Gamma + Inverse (典则连接)
*   **定义**: $V(\mu)=\mu^2, \quad \eta = 1/\mu, \quad g'(\mu) = -1/\mu^2 = -\eta^2$.
*   **IRLS 组件**:
    *   $W_{ii} = w_i \mu_i^2 = w_i \eta_i^{-2}$
    *   $z_i = \eta_i - \frac{y_i - \mu_i}{\mu_i^2}$
*   **注意**: 要求 $\eta > 0$ 约束。无 Line Search 极易发散。

---

## 5. 实现总结 (Implementation Summary)

1.  **IRLS 求解流程**:
    *   构建矩阵 $\mathbf{A} = \mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \hat{\phi} \mathbf{I}$.
    *   构建向量 $\mathbf{b} = \mathbf{X}^T \mathbf{W} \mathbf{z}$.
    *   求解线性方程 $\mathbf{A} \boldsymbol{\beta}_{new} = \mathbf{b}$.
2.  **数值稳定性 (Numerical Stability)**:
    *   对 $\eta$ (或 $\mu$) 进行截断 (Clamp)，防止 $e^\eta$ 溢出或除以零。
    *   对于 Tweedie 分布，在使用幂函数前确保底数为正。
