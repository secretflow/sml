# core 设计细化（link / distribution / family）

核心目标：提供数学原子，显式导数与方差函数，支持 canonical link 自动回退。

## Link 模块
接口契约：
```python
class Link(ABC):
    def link(self, mu): ...          # g(mu) -> eta
    def inverse(self, eta): ...       # g^{-1}(eta) -> mu
    def link_deriv(self, mu): ...     # d_eta / d_mu
    def inverse_deriv(self, eta): ... # d_mu / d_eta
```

内置实现（示例）：
- `IdentityLink`
- `LogLink`
- `LogitLink`
- `ProbitLink`
- `CLogLogLink`
- `PowerLink(power)`
- `ReciprocalLink`

数值要点：
- `inverse` 对 `eta` 做 clip（可由上层传入 clip_eta），避免 `exp` 溢出。
- `link` 对 `mu` 做 clip（clip_mu），避免 `log(0)`。

## Distribution 模块
接口契约：
```python
class Distribution(ABC):
    def unit_variance(self, mu): ...      # V(mu)
    def deviance(self, y, mu, weights=None): ...
    def starting_mu(self, y): ...         # 稳健初值
    def get_canonical_link(self) -> Link: ...
```

内置分布与 canonical link（建议）：
- Gaussian -> IdentityLink
- Bernoulli -> LogitLink
- Poisson -> LogLink
- Gamma -> LogLink
- Tweedie(p) -> LogLink（约定默认）
- NegativeBinomial -> LogLink

数值要点：
- `starting_mu`: 常用 `(y + mean(y)) / 2`，并 clip 正值区间。
- `unit_variance`: 显式公式，避免分段导数爆炸。
- `deviance`: 教科书定义，使用 `_clean` 避免 `log(0)`。

## Family 容器
职责：绑定 distribution 与 link；若 link 未提供，使用 `distribution.get_canonical_link()`。

```python
class Family:
    def __init__(self, distribution, link=None):
        self.distribution = distribution
        self.link = link if link is not None else distribution.get_canonical_link()
```

## 依赖与边界
- 不使用 `jax.grad` 等高阶算子，所有导数在 link/distribution 内显式实现。
- 对外仅暴露纯数学接口，业务逻辑（formula/solver）在上层完成。

## 扩展指引
- 新增 link：实现四个方法；若为某分布 canonical，更新对应分布的 `get_canonical_link`。
- 新增分布：实现四个方法；在工厂或注册表里登记 canonical link。