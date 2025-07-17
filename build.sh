#!/bin/bash
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

set -ex

detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "darwin"
        ;;
        Linux*)
            echo "linux"
        ;;
        CYGWIN* | MINGW* | MSYS*)
            echo "windows"
        ;;
        *)
            echo "unknown"
        ;;
    esac
}

detect_arch() {
    local arch=$(uname -m)
    case "$arch" in
        x86_64 | amd64)
            echo "x86_64"
        ;;
        i386 | i686)
            echo "i686"
        ;;
        aarch64 | arm64)
            echo "aarch64"
        ;;
        armv7l)
            echo "armv7l"
        ;;
        *)
            echo "$arch"
        ;;
    esac
}

get_macos_version() {
    if [ "$(detect_os)" = "darwin"]; then
        local version=$(sw_vers -productVersion)
        echo "$version" | sed 's/\([0-9]*\)\.\([0-9]*\).*/\1_\2/'
    else
        echo "10_9"
    fi
}

detect_platform() {
    local os=$(detect_os)
    local arch=$(detect_arch)
    
    case "$os" in
        windows)
            case "$arch" in
                x86_64)
                    echo "win_amd64"
                ;;
                aarch64)
                    echo "win_arm64"
                ;;
                *)
                    echo "win32"
                ;;
            esac
        ;;
        darwin)
            local mac_ver=$(get_macos_version)
            log_verbose "检测到macOS版本: $mac_ver"
            case "$arch" in
                aarch64)
                    echo "macosx_${mac_ver}_arm64"
                ;;
                *)
                    echo "macosx_${mac_ver}_x86_64"
                ;;
            esac
        ;;
        linux)
            case "$arch" in
                x86_64)
                    echo "linux_x86_64"
                ;;
                aarch64)
                    echo "linux_aarch64"
                ;;
                i686)
                    echo "linux_i686"
                ;;
                armv7l)
                    echo "linux_armv7l"
                ;;
                *)
                    echo "linux_${arch}"
                ;;
            esac
        ;;
        *)
            echo "any"
        ;;
    esac
}

detect_python_version() {
    local python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo $python_version | sed 's/\([0-9]*\)\.\([0-9]*\).*/\1\2/'
}

if [ -e sml/_version.py ]; then
    rm sml/_version.py
fi

platform_name=$(detect_platform)
echo "platform: $platform_name"
python_version=py$(detect_python_version)
echo "python: $python_version"

sed -i "s/^plat-name = .*/plat-name = $platform_name/" setup.cfg
sed -i "s/^python-tag = .*/python-tag = $python_version/" setup.cfg


SCRIPT_PATH=$(dirname "$0")
cd $SCRIPT_PATH

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$(pwd)
else
    export PYTHONPATH=$(pwd):$PYTHONPATH
fi

if python -m build --wheel; then
    exit 0
else
    exit 1
fi
