#! python3
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

import re
from datetime import datetime

from setuptools_scm.version import ScmVersion, simplified_semver_version


def timestamp_version_scheme(version: ScmVersion) -> str:
    version_str = simplified_semver_version(version)

    timestamp = datetime.now().strftime("%Y%m%d")

    return re.sub(r"dev\d+", f"dev{timestamp}", version_str)
