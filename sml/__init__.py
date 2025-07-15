# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import (
    cluster,
    decomposition,
    ensemble,
    feature_selection,
    gaussian_process,
    linear_model,
    metrics,
    naive_bayes,
    neighbors,
    preprocessing,
    svm,
    tree,
    utils,
)

try:
    from importlib.metadata import version

    __version__ = version("sf-sml")
except ImportError:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown"

__all__ = [
    "cluster",
    "decomposition",
    "ensemble",
    "feature_selection",
    "gaussian_process",
    "linear_model",
    "metrics",
    "naive_bayes",
    "neighbors",
    "preprocessing",
    "svm",
    "tree",
    "utils",
    "__version__",
]
