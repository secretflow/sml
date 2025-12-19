# SecretFlow SS-GLM 实现总结与分析

本文档总结了 `@secretflow/secretflow/ml/linear/ss_glm` 的设计与实现细节，并将其与 `sml/linear_model` 的现有设计进行对比。

## 1. 核心架构总结

SS-GLM (Secret Sharing Generalized Linear Model) 是 SecretFlow 中用于垂直联邦学习场景的 GLM 实现。它利用 SPU (Secure Processing Unit) 进行密态计算，支持 IRLS 和 SGD 两种求解器。

### 1.1 组件构成
- **`model.py`**: 核心逻辑，定义了 `SSGLM` 类，负责数据预处理、求解器调度 (IRLS/SGD)、模型评估与预测。
- **`core/link.py`**: 链接函数定义 (`Linker` 协议)，包含 Logit, Log, Reciprocal, Identity。
- **`core/distribution.py`**: 分布定义 (`Distribution` 协议)，包含 Bernoulli, Poisson, Gamma, Tweedie。
- **`metrics.py`**: 评估指标实现 (Deviance, AUC, RMSE 等)。

### 1.2 数据流与隐私策略
- **输入数据**: 支持 `FedNdarray` (垂直切分数组) 和 `VDataFrame`。
- **计算模式**: 大部分计算在 SPU 中进行（密态）。
- **信息泄露 (Reveal) 情况**:
    - **Hessian 矩阵 (IRLS)**: 在 IRLS 求解中，Hessian 矩阵 $J = X^T W X$ 被显式 Reveal 给 Label Holder (`self.y_device`) 进行求逆。这是一个为了性能和数值稳定性的权衡（SPU 内求逆代价高昂且不稳定）。
    - **初始权重 (IRLS)**: 首个 Epoch 的初始权重 $W$ 和工作响应 $z$ 是在 Label Holder 本地（明文）计算的。
    - **评估指标**: 训练过程中的停止指标（如 Deviance, Change Rate）会被 Reveal 以进行 Early Stopping 判断。
    - **Y Scale**: 目标变量的缩放因子会被 Reveal。

---

## 2. 数学公式与实现对比

SS-GLM 的数学实现与 `sml/linear_model` 的设计 (**Canonical GLM**) 高度一致。

### 2.1 符号对照
| 概念 | 数学符号 | SS-GLM 代码 | SML Design |
| :--- | :--- | :--- | :--- |
| 链接函数 | $g(\mu) = \eta$ | `Linker.link` | `Link.link` |
| 逆链接 | $\mu = g^{-1}(\eta)$ | `Linker.response` | `Link.inverse` |
| 链接导数 | $g'(\mu)$ | `Linker.link_derivative` | `Link.link_deriv` |
| 响应导数 | $d\mu/d\eta = 1/g'$ | `Linker.response_derivative` | `Link.inverse_deriv` |
| 方差函数 | $V(\mu)$ | `Distribution.variance` | `Distribution.unit_variance` |

### 2.2 IRLS 求解器对比
**SS-GLM 实现**:
- **权重**: `W_diag = 1 / dist.scale() / (v * g_gradient) / g_gradient`
  $$ W = \frac{1}{\phi V(\mu) (g'(\mu))^2} $$
- **工作响应**: `Z = eta + (y - mu) * g_gradient`
  $$ z = \eta + (y - \mu) g'(\mu) $$
- **更新**: 使用明文求逆 `inv_J = y_device(J_inv)(J)`，然后密态更新 `beta = inv_J @ XTWZ`。

**SML Design**:
- **公式**: 完全一致。
- **差异**: SML Design 采用 `formula` 抽象，且当前设计为 JAX 通用实现，使用 `invert_matrix` (Naive Inv) 但未强制指定 Reveal 策略（由后端决定）。SS-GLM 明确指定了 Reveal 策略。

### 2.3 SGD 求解器对比
**SS-GLM 实现**:
- **梯度计算**:
  `grad = link.response_derivative(pred)` ($1/g'$) 
  `temp = grad * err / dev` ($ \frac{\mu - y}{V g'} $) 
  `devp = X.T @ temp` ($ X^T \frac{\mu - y}{V g'} $) 
  $$ \nabla_{NLL} = X^T \frac{\mu - y}{V(\mu) g'(\mu)} $$
- **正则化**: 手动添加 L2 梯度。

**SML Design**:
- **公式**: 利用 `Formula` 组件 $W \cdot z_{resid} = \frac{y-\mu}{V g'}$，梯度为 $X^T (W \cdot z_{resid})$。
- **一致性**: 数学上完全等价。SS-GLM 的 `temp` 变量即对应 SML Design 中的 `W * z_resid` (符号相反，取决于优化目标是 LL 还是 NLL)。

---

## 3. 实现细节总结

### 3.1 Link 与 Distribution
SS-GLM 的 `core` 模块实现了一组标准的 GLM 组件。
- **Linker**: 包含 Logit, Log, Reciprocal, Identity。
- **Distribution**: 包含 Bernoulli, Poisson, Gamma, Tweedie。
- **特别处理**: `Tweedie` 分布根据 `power` 参数动态调整方差公式，包含 0 (Normal), 1 (Poisson), 2 (Gamma) 及 (1, 2) 之间的混合态。

### 3.2 优化与工程特性
- **SPU Cache**: 使用 `spu.experimental.make_cached_var` 对 `g_gradient`, `XTW` 等中间变量进行缓存，减少重复通信。
- **Batching**: 支持 SGD 的 Batch 训练，使用了 `_build_batch_cache` 机制来处理垂直切分数据的 Batch 读取。
- **Early Stopping**: 基于验证集指标（Deviance, AUC 等）或权重变化率 (`weight change rate`)。
- **Data Scaling**: 对 $y$ 进行了缩放处理 (`y_scale`) 以提升数值稳定性。

### 3.3 差异点与借鉴
- **Formula 抽象**: SS-GLM 没有显式的 `Formula` 抽象，逻辑散落在 `_irls_calculate_partials` 和 `_sgd_update_w` 中。SML Design 的 `Formula` 模式更利于扩展手写优化。
- **Reveal 策略**: SS-GLM 在 IRLS 中 Reveal Hessian 是一个关键的工程决策，这在纯 MPC 环境下通常是必须的。SML Design 在未来适配 MPC 后端时应参考此策略。

## 4. 结论
`secretflow/secretflow/ml/linear/ss_glm` 的实现逻辑严谨，数学推导正确，与 `sml/linear_model` 的新设计在理论层面完全一致。其处理 MPC 场景下的特有技巧（如 Hessian Reveal、数据分片缓存）对于未来 SML 库在 MPC 后端的落地具有重要的参考价值。
