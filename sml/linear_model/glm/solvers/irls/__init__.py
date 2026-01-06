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

"""IRLS solvers including generic and optimized implementations."""

from sml.linear_model.glm.solvers.irls_generic import IRLSSolver

# Import optimized solvers
from .bernoulli_logit import BernoulliLogitIRLSSolver
from .gamma_inverse import GammaInverseIRLSSolver
from .gamma_log import GammaLogIRLSSolver
from .gaussian_identity import GaussianIdentityIRLSSolver
from .poisson_log import PoissonLogIRLSSolver
from .tweedie_log import TweedieLogIRLSSolver

__all__ = [
    "IRLSSolver",
    "GaussianIdentityIRLSSolver",
    "PoissonLogIRLSSolver",
    "BernoulliLogitIRLSSolver",
    "GammaLogIRLSSolver",
    "GammaInverseIRLSSolver",
    "TweedieLogIRLSSolver",
]
