这是一份关于 **广义线性模型（GLM）算法核心计算手册**。

这份报告采用了严格的矩阵记号，详细推导了 **IRLS（二阶优化）** 和 **SGD（一阶优化）** 在包含 **离散参数（Dispersion Parameter $\\phi$）**、**样本权重（Sample Weights $\\boldsymbol{w}$）** 以及 **L2 正则化** 时的完整迭代公式。

此外，针对业务中高频使用的特定分布与连接函数组合，推导并化简了其特有的矩阵运算形式，以供算法库开发时直接使用，避免通用自动微分带来的数值不稳定。

# ---

**广义线性模型 (GLM) 计算与算法设计手册**

## **1\. 符号体系定义 (Notation System)**

为了确保公式的严谨性，全文统一使用以下矩阵符号。

**基本维度：**

* $N$ : 样本数量 (Number of samples)  
* $p$ : 特征数量 (Number of features, 包含截距项)

**数据与参数矩阵：**

| 符号 | 维度 | 含义 | 备注 |
| :---- | :---- | :---- | :---- |
| $\\mathbf{X}$ | $N \\times p$ | 设计矩阵 (Design Matrix) | 第 $i$ 行记为 $\\mathbf{x}\_i^T$ |
| $\\mathbf{y}$ | $N \\times 1$ | 响应变量向量 (Response) | $\\mathbf{y} \= \[y\_1, \\dots, y\_N\]^T$ |
| $\\boldsymbol{\\beta}$ | $p \\times 1$ | 回归系数向量 (Coefficients) | 待优化参数 |
| $\\mathbf{w}$ | $N \\times 1$ | 样本权重向量 (Sample Weights) | 默认为全 1 向量 |
| $\\boldsymbol{\\eta}$ | $N \\times 1$ | 线性预测子 (Linear Predictor) | $\\boldsymbol{\\eta} \= \\mathbf{X}\\boldsymbol{\\beta} \+ \\text{offset}$ |
| $\\boldsymbol{\\mu}$ | $N \\times 1$ | 期望均值向量 (Mean) | $\\mu\_i \= E\[y\_i |

**函数与统计量：**

| 符号 | 类型 | 含义 | 数学定义 |
| :---- | :---- | :---- | :---- |
| $g(\\cdot)$ | Element-wise | 连接函数 (Link Function) | $\\boldsymbol{\\eta} \= g(\\boldsymbol{\\mu})$ |
| $g'(\\cdot)$ | Element-wise | 连接函数一阶导数 | $g'(\\mu) \= \\frac{d\\eta}{d\\mu}$ |
| $V(\\cdot)$ | Element-wise | 方差函数 (Variance Function) | $\\text{Var}(y\_i) \= \\frac{\\phi}{w\_i} V(\\mu\_i)$ |
| $\\phi$ | 标量 | **离散参数 (Dispersion)** | 控制分布的方差缩放 |
| $\\lambda$ | 标量 | L2 正则化系数 | 惩罚项强度 |
| $\\alpha$ | 标量 | 学习率 (Learning Rate) | 仅用于 SGD |

## ---

**2\. 优化目标函数 (Objective Function)**

GLM 的目标是最大化对数似然函数，或者等价地，最小化 **带 L2 正则化的负对数似然 (Regularized Negative Log-Likelihood, NLL)**。

考虑指数族分布的通用形式，目标函数 $J(\\boldsymbol{\\beta})$ 定义为：

$$J(\\boldsymbol{\\beta}) \= \\underbrace{- \\sum\_{i=1}^N \\frac{w\_i}{\\phi} \\left( y\_i \\theta\_i \- b(\\theta\_i) \\right)}\_{\\text{Negative Log-Likelihood}} \+ \\underbrace{\\frac{\\lambda}{2} \\|\\boldsymbol{\\beta}\\|\_2^2}\_{\\text{L2 Penalty}}$$

其中 $\\theta\_i$ 是典则参数 (Canonical Parameter)，满足 $b'(\\theta\_i) \= \\mu\_i$ 且 $b''(\\theta\_i) \= V(\\mu\_i)$。连接函数建立了 $\\mu\_i$ 与 $\\mathbf{x}\_i^T \\boldsymbol{\\beta}$ 的关系。

**注意**：离散参数 $\\phi$ 出现在似然项的分母中。在优化过程中，为了数值计算的一致性，$\\phi$ 的存在会影响正则化项的相对权重。

## ---

**3\. 通用计算框架 (General Framework)**

本节给出适用于任意 GLM 组合的通用矩阵更新公式。

### **3.1 离散参数 $\\phi$ 的估计**

在 $\\boldsymbol{\\beta}$ 的迭代过程中，$\\phi$ 通常被视为常数。每当 $\\boldsymbol{\\beta}$ 收敛或更新一轮后，使用 **皮尔逊卡方估计量 (Pearson Chi-squared Estimator)** 更新 $\\phi$：

$$\\hat{\\phi} \= \\frac{1}{N \- p} \\sum\_{i=1}^N \\frac{w\_i (y\_i \- \\mu\_i)^2}{V(\\mu\_i)}$$

* 若分布为 Poisson 或 Binomial，理论上 $\\phi=1$。但计算此值可用于检测数据是否存在“过离散 (Over-dispersion)”现象。

### ---

**3.2 算法一：IRLS (迭代加权最小二乘)**

IRLS 本质上是 **牛顿-拉夫逊法 (Newton-Raphson)** 的一种特殊形式，利用二阶海森矩阵信息进行快速收敛。

#### **核心组件矩阵构建**

在第 $t$ 次迭代中，基于当前的 $\\boldsymbol{\\beta}^{(t)}$ 计算 $\\boldsymbol{\\eta}$ 和 $\\boldsymbol{\\mu}$，然后构建以下两个关键矩阵：

1. 工作权重矩阵 (Working Weight Matrix) $\\mathbf{W}$  
   这是一个 $N \\times N$ 的对角矩阵：  
   $$\\mathbf{W} \= \\text{diag}\\left( \\frac{w\_i}{V(\\mu\_i) \\cdot \[g'(\\mu\_i)\]^2} \\right)$$  
2. 工作响应向量 (Working Response Vector) $\\mathbf{z}$  
   这是一个 $N \\times 1$ 向量，它是响应变量在当前预测值处的一阶泰勒展开：  
   $$\\mathbf{z} \= \\boldsymbol{\\eta} \+ \\text{diag}(g'(\\boldsymbol{\\mu})) (\\mathbf{y} \- \\boldsymbol{\\mu})$$  
   元素形式：$z\_i \= \\eta\_i \+ (y\_i \- \\mu\_i) g'(\\mu\_i)$

#### **参数更新公式 (Matrix Update Rule)**

标准的 Newton 更新步为 $\\boldsymbol{\\beta}^{(t+1)} \= \\boldsymbol{\\beta}^{(t)} \- \\mathbf{H}^{-1} \\nabla J$。经过推导与化简（消除 $\\phi$ 分母），得到以下加权最小二乘形式：

$$\\left( \\mathbf{X}^T \\mathbf{W} \\mathbf{X} \+ \\lambda \\hat{\\phi} \\mathbf{I} \\right) \\boldsymbol{\\beta}^{(t+1)} \= \\mathbf{X}^T \\mathbf{W} \\mathbf{z}$$

**公式解析与实现细节**：

* **左侧 (Hessian 近似)**: $\\mathbf{X}^T \\mathbf{W} \\mathbf{X}$ 是 Fisher Information Matrix 的一部分。注意正则化项变为 $\\lambda \\hat{\\phi} \\mathbf{I}$。这是因为原始似然函数的 Hessian 包含 $1/\\phi$ 因子，为了将线性方程组转化为标准 $Ax=b$ 形式，方程两边同乘了 $\\phi$。  
* **右侧 (Target)**: $\\mathbf{X}^T \\mathbf{W} \\mathbf{z}$。**绝对不要**在此处减去 $\\lambda \\boldsymbol{\\beta}$。正则化梯度的影响已经完全由左侧矩阵中的 $\\lambda \\hat{\\phi} \\mathbf{I}$ 隐式处理（通过 $\\boldsymbol{\\beta}^{(t)}$ 的展开与消元）。  
* **求解**: 使用 Cholesky 分解或 QR 分解求解该线性方程组，避免直接求逆。

### ---

**3.3 算法二：SGD (随机梯度下降)**

当 $N$ 非常大无法存储 $\\mathbf{X}^T \\mathbf{W} \\mathbf{X}$ ($p \\times p$ 稠密矩阵) 时，使用 SGD。SGD 仅利用一阶梯度。

#### **梯度计算公式**

总目标函数 $J(\\boldsymbol{\\beta})$ 关于 $\\boldsymbol{\\beta}$ 的梯度为：

$$\\nabla J(\\boldsymbol{\\beta}) \= \-\\frac{1}{\\phi} \\mathbf{X}^T \\left\[ \\mathbf{w} \\odot \\frac{\\mathbf{y} \- \\boldsymbol{\\mu}}{V(\\boldsymbol{\\mu}) \\odot g'(\\boldsymbol{\\mu})} \\right\] \+ \\lambda \\boldsymbol{\\beta}$$

*注：$\\odot$ 表示逐元素相乘，除法也是逐元素相除。*

为了简化计算，定义 **辅助因子向量 (Auxiliary Vector) $\\mathbf{c}$**，维度 $N \\times 1$：

$$c\_i \= \\frac{w\_i}{V(\\mu\_i) \\cdot g'(\\mu\_i)}$$

则梯度可简写为：

$$\\nabla J(\\boldsymbol{\\beta}) \= \-\\frac{1}{\\phi} \\mathbf{X}^T (\\mathbf{c} \\odot (\\mathbf{y} \- \\boldsymbol{\\mu})) \+ \\lambda \\boldsymbol{\\beta}$$

#### **参数更新公式**

采用 **权重衰减 (Weight Decay)** 形式的更新规则：

$$\\boldsymbol{\\beta}^{(t+1)} \= (1 \- \\alpha \\lambda) \\boldsymbol{\\beta}^{(t)} \+ \\frac{\\alpha}{\\hat{\\phi}} \\cdot \\mathbf{X}\_{batch}^T \\left( \\mathbf{c}\_{batch} \\odot (\\mathbf{y}\_{batch} \- \\boldsymbol{\\mu}\_{batch}) \\right)$$

**实现差异**：与 IRLS 不同，SGD 必须显式计算正则化梯度 $\\lambda \\boldsymbol{\\beta}$ (体现为权重衰减项 $(1-\\alpha\\lambda)$)。

## ---

**4\. 常见分布的特化公式 (Specialized Formula Cheatsheet)**

为了数值稳定性与计算效率，**严禁**在实现特定组合时直接调用通用的 variance() 和 derivative() 函数进行链式计算。必须使用下方化简后的代数形式构建矩阵 $\\mathbf{W}$ 和向量 $\\mathbf{z}$。

### **4.1 高斯分布 (Linear Regression)**

* **组合**: Gaussian Distribution \+ Identity Link  
* **定义**: $V(\\mu)=1, \\quad \\eta=\\mu, \\quad g'(\\mu)=1$  
* **IRLS 矩阵组件**:  
  * $\\mathbf{W} \= \\text{diag}(\\mathbf{w})$  
  * $\\mathbf{z} \= \\mathbf{y}$  
* 更新公式:

  $$(\\mathbf{X}^T \\text{diag}(\\mathbf{w}) \\mathbf{X} \+ \\lambda \\hat{\\sigma}^2 \\mathbf{I}) \\boldsymbol{\\beta} \= \\mathbf{X}^T (\\mathbf{w} \\odot \\mathbf{y})$$

### **4.2 伯努利分布 (Logistic Regression)**

* **组合**: Bernoulli Distribution \+ Logit Link (Canonical)  
* **定义**:  
  * $V(\\mu) \= \\mu(1-\\mu)$  
  * $\\mu \= \\frac{1}{1 \+ e^{-\\eta}}$ (Sigmoid)  
  * $g'(\\mu) \= \\frac{1}{\\mu(1-\\mu)}$  
* **IRLS 矩阵组件 (化简后)**:  
  * $\\mathbf{W}\_{ii} \= w\_i \\cdot \\mu\_i (1 \- \\mu\_i)$  
    * *解释*: $1 / \[\\mu(1-\\mu) \\cdot (\\frac{1}{\\mu(1-\\mu)})^2\] \= \\mu(1-\\mu)$  
  * $z\_i \= \\eta\_i \+ \\frac{y\_i \- \\mu\_i}{\\mu\_i (1-\\mu\_i)}$  
* **SGD 梯度项**:  
  * $\\mathbf{c} \= \\mathbf{w}$ (辅助因子为样本权重本身)  
  * $\\nabla\_{data} \= \-\\mathbf{X}^T (\\mathbf{w} \\odot (\\mathbf{y} \- \\boldsymbol{\\mu}))$

### **4.3 泊松分布 (Poisson Regression)**

* **组合**: Poisson Distribution \+ Log Link (Canonical)  
* **定义**:  
  * $V(\\mu) \= \\mu$  
  * $\\mu \= e^{\\eta}$  
  * $g'(\\mu) \= \\frac{1}{\\mu}$  
* **IRLS 矩阵组件 (化简后)**:  
  * $\\mathbf{W}\_{ii} \= w\_i \\cdot \\mu\_i$  
    * *解释*: $1 / \[\\mu \\cdot (1/\\mu)^2\] \= \\mu$  
  * $z\_i \= \\eta\_i \+ \\frac{y\_i \- \\mu\_i}{\\mu\_i} \= \\eta\_i \+ \\frac{y\_i}{\\mu\_i} \- 1$  
* **SGD 梯度项**:  
  * $\\mathbf{c} \= \\mathbf{w}$  
  * $\\nabla\_{data} \= \-\\mathbf{X}^T (\\mathbf{w} \\odot (\\mathbf{y} \- \\boldsymbol{\\mu}))$

### **4.4 Gamma 分布 (Log Link) —— 业务常用**

* **组合**: Gamma Distribution \+ Log Link (非典则，但保证 $\\mu\>0$)  
* **定义**:  
  * $V(\\mu) \= \\mu^2$  
  * $\\mu \= e^{\\eta}$  
  * $g'(\\mu) \= \\frac{1}{\\mu}$  
* **IRLS 矩阵组件 (极简形式)**:  
  * $\\mathbf{W}\_{ii} \= w\_i$ (**常数权重\!**)  
    * *推导*: $W \= 1 / \[\\mu^2 \\cdot (1/\\mu)^2\] \= 1$。这意味着 Hessian 的曲率部分不随预测值变化，数值非常稳定。  
  * $z\_i \= \\eta\_i \+ \\frac{y\_i \- \\mu\_i}{\\mu\_i} \= \\eta\_i \+ \\frac{y\_i}{\\mu\_i} \- 1$  
* **SGD 梯度项**:  
  * $c\_i \= w\_i / \\mu\_i$  
  * $\\nabla\_{data} \= \-\\mathbf{X}^T (\\mathbf{w} \\odot \\frac{\\mathbf{y} \- \\boldsymbol{\\mu}}{\\boldsymbol{\\mu}})$

### **4.5 Tweedie 分布 (Log Link) —— 保险定价核心**

* **组合**: Tweedie ($p \\in (1, 2)$) \+ Log Link  
* **定义**:  
  * $V(\\mu) \= \\mu^p$  
  * $\\mu \= e^{\\eta}$  
  * $g'(\\mu) \= \\frac{1}{\\mu}$  
* **IRLS 矩阵组件 (需手写优化)**:  
  * $\\mathbf{W}\_{ii} \= w\_i \\cdot \\mu\_i^{2-p}$  
    * *实现建议*: 代码中直接计算 $w\_i \\cdot \\exp(\\eta\_i \\cdot (2-p))$。避免先算 $\\mu^p$ 再除以 $\\mu^2$ 导致溢出。  
  * $z\_i \= \\eta\_i \+ y\_i e^{-\\eta\_i} \- 1$  
* **SGD 梯度项**:  
  * $c\_i \= w\_i \\cdot \\mu\_i^{1-p}$  
  * $\\nabla\_{data} \= \-\\mathbf{X}^T \\left( \\mathbf{w} \\odot (\\mathbf{y} \- \\boldsymbol{\\mu}) \\odot e^{\\boldsymbol{\\eta}(1-p)} \\right)$

### **4.6 Gamma 分布 (Inverse Link) —— 典则连接**

* **组合**: Gamma Distribution \+ Inverse Link  
* **定义**:  
  * $V(\\mu) \= \\mu^2$  
  * $\\eta \= \\mu^{-1} \\implies \\mu \= \\eta^{-1}$  
  * $g'(\\mu) \= \-\\mu^{-2} \= \-\\eta^2$  
* **IRLS 矩阵组件**:  
  * $\\mathbf{W}\_{ii} \= w\_i \\cdot \\mu\_i^2 \= w\_i \\cdot \\eta\_i^{-2}$  
  * $z\_i \= \\eta\_i \- \\frac{y\_i \- \\mu\_i}{\\mu\_i^2}$  
* **注意**: 逆连接函数要求 $\\eta$ 必须始终非零且保持符号一致（通常为正），数值上极易不稳定，需配合 Line Search 或 Bound Constraint。

## ---

**5\. 实现建议与总结**

1. **IRLS 求解器**:  
   * 核心在于构建 $\\mathbf{X}^T \\mathbf{W} \\mathbf{X} \+ \\lambda \\hat{\\phi} \\mathbf{I}$。  
   * 对于 **Gamma+Log**，由于 $\\mathbf{W}$ 是常数（仅依赖样本权重），Hessian 矩阵 $\\mathbf{X}^T \\mathbf{W} \\mathbf{X}$ 在整个训练过程中是**固定**的（如果不考虑 $\\phi$ 的变化）。可以缓存该矩阵分解结果以极大加速迭代。  
2. **数值保护**:  
   * 在计算 $\\mu \= e^\\eta$ (Log Link) 或 $\\mu \= 1/\\eta$ (Inverse Link) 时，务必对 $\\eta$ 进行截断 (Clipping)，防止 $\\mu$ 溢出或趋近于 0。  
   * Tweedie 分布中计算 $\\mu^{2-p}$ 时，确保底数为正。  
3. **正则化修正**:  
   * 再次强调，在 IRLS 的 Linear Solver 步骤中，右侧目标向量 $\\mathbf{X}^T \\mathbf{W} \\mathbf{z}$ **不包含**正则化项。正则化完全由左侧矩阵的 $\\lambda \\hat{\\phi} \\mathbf{I}$ 承担。