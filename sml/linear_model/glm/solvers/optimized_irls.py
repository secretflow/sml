# Copyright 2025 Ant Group Co., Ltd.
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

from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.core.family import Family

_REGISTRY: dict[Family, Solver] = {}


def register_solver(family: Family, solver: Solver):
    """
    Register an optimized solver for a specific family.

    Parameters
    ----------
    family : Family
        The GLM family (distribution + link).
    solver : Solver
        The solver implementation to use.
    """
    _REGISTRY[family] = solver


def get_registered_solver(family: Family) -> Solver | None:
    """
    Retrieve the registered solver for a specific family.

    Parameters
    ----------
    family : Family
        The GLM family (distribution + link).

    Returns
    -------
    Solver or None
        The registered solver if found, otherwise None.
    """
    return _REGISTRY.get(family, None)
