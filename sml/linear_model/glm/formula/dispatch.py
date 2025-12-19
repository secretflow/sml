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


from sml.linear_model.glm.core.distribution import Distribution
from sml.linear_model.glm.core.link import Link

from .base import Formula
from .generic import GenericFormula


class FormulaDispatcher:
    """
    Registry for GLM formulas.

    Dispatches to optimized implementations based on (distribution_type, link_type).
    Falls back to GenericFormula if no optimized implementation is found.
    """

    def __init__(self):
        # Registry: (DistType, LinkType) -> Formula Class or Instance
        self._registry: dict[tuple[type[Distribution], type[Link]], Formula] = {}

    def register(
        self, dist_type: type[Distribution], link_type: type[Link], formula: Formula
    ):
        """
        Register an optimized formula for a specific distribution and link pair.

        Parameters
        ----------
        dist_type : Type[Distribution]
            The class of the distribution.
        link_type : Type[Link]
            The class of the link function.
        formula : Formula
            The formula implementation to use.
        """
        self._registry[(dist_type, link_type)] = formula

    def resolve(self, distribution: Distribution, link: Link) -> Formula:
        """
        Find the best formula implementation for the given distribution and link.

        Parameters
        ----------
        distribution : Distribution
            Instance of the distribution.
        link : Link
            Instance of the link function.

        Returns
        -------
        formula : Formula
            The resolved formula implementation.
        """
        key = (type(distribution), type(link))
        # 1. Try to find optimized formula in registry
        if key in self._registry:
            return self._registry[key]

        # 2. Fallback to GenericFormula
        return GenericFormula()


# Global dispatcher instance
dispatcher = FormulaDispatcher()


def register_formula(dist_type: type[Distribution], link_type: type[Link], formula: Formula):
    """Global helper to register a formula."""
    dispatcher.register(dist_type, link_type, formula)
