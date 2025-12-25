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

"""
Registry for optimized IRLS solvers.

This module provides a registry mechanism to map specific GLM families
(distribution + link combinations) to their optimized solver implementations.

Usage:
------
1. Register a solver:
   >>> from sml.linear_model.glm.solvers.optimized_irls import register_solver
   >>> register_solver(family, solver_instance)

2. Get a registered solver:
   >>> solver = get_registered_solver(family)
   >>> if solver is not None:
   ...     # Use optimized solver
   ... else:
   ...     # Fall back to generic solver
"""

from sml.linear_model.glm.core.distribution import Gamma, Tweedie
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.core.link import LogLink
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.irls import GammaLogIRLSSolver, TweedieLogIRLSSolver

_REGISTRY: dict[str, Solver] = {}


def _make_family_key(family: Family) -> str:
    """
    Create a unique string key for a family.

    Uses distribution and link class names to create the key.
    For distributions with parameters (like Tweedie), includes the parameter.
    """
    dist_name = family.distribution.__class__.__name__
    link_name = family.link.__class__.__name__

    # Handle parameterized distributions
    if isinstance(family.distribution, Tweedie):
        dist_name = f"Tweedie(p={family.distribution.power})"

    return f"{dist_name}+{link_name}"


def register_solver(family: Family, solver: Solver) -> None:
    """
    Register an optimized solver for a specific family.

    Parameters
    ----------
    family : Family
        The GLM family (distribution + link).
    solver : Solver
        The solver implementation to use.
    """
    key = _make_family_key(family)
    _REGISTRY[key] = solver


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
    key = _make_family_key(family)
    return _REGISTRY.get(key, None)


def register_default_optimized_solvers() -> None:
    """
    Register default optimized solvers for common family combinations.

    Currently registers:
    - Gamma + Log: Constant working weights optimization
    - Tweedie + Log: Direct exp(eta*(2-p)) computation
    """
    # Gamma + Log
    gamma_log_family = Family(Gamma(), LogLink())
    register_solver(gamma_log_family, GammaLogIRLSSolver())

    # Tweedie + Log with common power values
    for power in [1.5]:  # Add more power values as needed
        tweedie_log_family = Family(Tweedie(power=power), LogLink())
        register_solver(tweedie_log_family, TweedieLogIRLSSolver(power=power))


def list_registered_solvers() -> list[str]:
    """
    List all registered solver keys.

    Returns
    -------
    list[str]
        List of registered family keys.
    """
    return list(_REGISTRY.keys())


# Auto-register default optimized solvers on module import
register_default_optimized_solvers()
