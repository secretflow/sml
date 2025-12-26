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

The registry uses a two-level matching strategy:
1. First try exact match (distribution class + link class)
2. For parameterized distributions like Tweedie, create solver with matched parameters

Usage:
------
1. Get a registered solver (automatically handles Tweedie power):
   >>> solver = get_registered_solver(family)
   >>> if solver is not None:
   ...     # Use optimized solver
   ... else:
   ...     # Fall back to generic solver

2. Check available solvers:
   >>> print(list_registered_solvers())
"""

from collections.abc import Callable

from sml.linear_model.glm.core.distribution import (
    Bernoulli,
    Distribution,
    Gamma,
    Normal,
    Poisson,
    Tweedie,
)
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.core.link import (
    IdentityLink,
    LogitLink,
    LogLink,
    ReciprocalLink,
)
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.irls import (
    BernoulliLogitIRLSSolver,
    GammaInverseIRLSSolver,
    GammaLogIRLSSolver,
    GaussianIdentityIRLSSolver,
    PoissonLogIRLSSolver,
    TweedieLogIRLSSolver,
)

# Type for solver factory functions
SolverFactory = Callable[[Distribution], Solver]


# Registry stores either a Solver instance or a factory function
_REGISTRY: dict[str, Solver | SolverFactory] = {}


def _make_family_key(dist_type: type, link_type: type) -> str:
    """
    Create a unique string key for a family based on types.

    Parameters
    ----------
    dist_type : type
        The distribution class (e.g., Normal, Tweedie).
    link_type : type
        The link function class (e.g., LogLink, IdentityLink).

    Returns
    -------
    str
        A unique key like "Normal+IdentityLink".
    """
    return f"{dist_type.__name__}+{link_type.__name__}"


def register_solver(
    dist_type: type,
    link_type: type,
    solver_or_factory: Solver | SolverFactory,
) -> None:
    """
    Register an optimized solver for a specific distribution+link combination.

    Parameters
    ----------
    dist_type : type
        The distribution class (e.g., Normal, Tweedie).
    link_type : type
        The link function class (e.g., LogLink, IdentityLink).
    solver_or_factory : Solver | SolverFactory
        Either a Solver instance (for simple cases) or a factory function
        that takes a Distribution and returns a Solver (for parameterized
        distributions like Tweedie).
    """
    key = _make_family_key(dist_type, link_type)
    _REGISTRY[key] = solver_or_factory


def get_registered_solver(family: Family) -> Solver | None:
    """
    Retrieve the registered solver for a specific family.

    This function handles parameterized distributions (like Tweedie) by
    calling the factory function with the actual distribution instance
    to extract parameters.

    Parameters
    ----------
    family : Family
        The GLM family (distribution + link).

    Returns
    -------
    Solver or None
        The registered solver if found, otherwise None.
    """
    dist_type = type(family.distribution)
    link_type = type(family.link)
    key = _make_family_key(dist_type, link_type)

    solver_or_factory = _REGISTRY.get(key, None)
    if solver_or_factory is None:
        return None

    # If it's a factory function (callable but not a solver object),
    # call it with the distribution to get the solver.
    # We check if it has a 'solve' method to determine if it's a Solver.
    if callable(solver_or_factory) and not hasattr(solver_or_factory, "solve"):
        return solver_or_factory(family.distribution)

    # At this point, solver_or_factory is a Solver (has 'solve' method)
    return solver_or_factory  # type: ignore[return-value]


def _tweedie_log_solver_factory(dist: Distribution) -> Solver:
    """
    Factory function to create TweedieLogIRLSSolver with the correct power.

    Parameters
    ----------
    dist : Distribution
        The Tweedie distribution instance.

    Returns
    -------
    Solver
        A TweedieLogIRLSSolver with matching power parameter.
    """
    if not isinstance(dist, Tweedie):
        raise TypeError(f"Expected Tweedie distribution, got {type(dist).__name__}")
    return TweedieLogIRLSSolver(power=dist.power)


def register_default_optimized_solvers() -> None:
    """
    Register default optimized solvers for common family combinations.

    Currently registers:
    - Gaussian + Identity: One-iteration weighted least squares
    - Poisson + Log: Canonical link optimization
    - Bernoulli + Logit: Canonical link optimization
    - Gamma + Log: Constant working weights optimization
    - Gamma + Inverse: Canonical link (warning: numerically challenging)
    - Tweedie + Log: Direct exp(eta*(2-p)) computation (any power)
    """
    # Gaussian + Identity (Linear Regression)
    register_solver(Normal, IdentityLink, GaussianIdentityIRLSSolver())

    # Poisson + Log (Canonical)
    register_solver(Poisson, LogLink, PoissonLogIRLSSolver())

    # Bernoulli + Logit (Logistic Regression, Canonical)
    register_solver(Bernoulli, LogitLink, BernoulliLogitIRLSSolver())

    # Gamma + Log
    register_solver(Gamma, LogLink, GammaLogIRLSSolver())

    # Gamma + Inverse (Canonical, numerically challenging)
    register_solver(Gamma, ReciprocalLink, GammaInverseIRLSSolver())

    # Tweedie + Log (uses factory to handle any power value)
    register_solver(Tweedie, LogLink, _tweedie_log_solver_factory)


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
