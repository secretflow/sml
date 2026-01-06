# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .distribution import Distribution
from .link import Link


class Family:
    """
    A container that binds a Distribution and a Link function.

    The Family class encapsulates the statistical assumptions of the GLM.
    It automatically handles the default canonical link if one is not provided.

    Parameters
    ----------
    distribution : Distribution
        The exponential family distribution (e.g., Normal, Poisson, Bernoulli).
    link : Link, optional
        The link function. If None, the distribution's canonical link is used.
    """

    def __init__(self, distribution: Distribution, link: Link | None = None):
        self.distribution = distribution
        if link is None:
            self.link = distribution.get_canonical_link()
        else:
            self.link = link

    def __repr__(self) -> str:
        """Return a string representation of the Family."""
        return f"Family(distribution={self.distribution.__class__.__name__}, link={self.link.__class__.__name__})"
