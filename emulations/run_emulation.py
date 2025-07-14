#!/usr/bin/env python3
# Copyright 2024 Ant Group Co., Ltd.
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

"""
Main runner script for SML emulations.

This script provides a convenient way to run emulation tests for different modules.
"""

import argparse
import importlib
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sml.utils.emulation as emulation


def list_emulations():
    """List all available emulation modules."""
    emulations_dir = Path(__file__).parent
    emulation_files = []

    for module_dir in emulations_dir.iterdir():
        if module_dir.is_dir() and module_dir.name != "__pycache__":
            for emul_file in module_dir.glob("*_emul.py"):
                module_path = f"emulations.{module_dir.name}.{emul_file.stem}"
                emulation_files.append(module_path)

    return sorted(emulation_files)


def run_emulation(module_path: str, mode: str = "multiprocess"):
    """Run a specific emulation module."""
    try:
        module = importlib.import_module(module_path)

        # Convert mode string to emulation.Mode enum
        if mode.lower() == "multiprocess":
            emul_mode = emulation.Mode.MULTIPROCESS
        elif mode.lower() == "docker":
            emul_mode = emulation.Mode.DOCKER
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Look for the main emulation function (usually starts with "emul_")
        emul_functions = [name for name in dir(module) if name.startswith("emul_")]

        if not emul_functions:
            print(f"No emulation function found in {module_path}")
            return False

        # Run the first emulation function found
        emul_func = getattr(module, emul_functions[0])
        print(f"Running {module_path}.{emul_functions[0]} in {mode} mode...")
        emul_func(emul_mode)
        return True

    except Exception as e:
        print(f"Error running {module_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run SML emulation tests")
    parser.add_argument(
        "module",
        nargs="?",
        help="Emulation module to run (e.g., emulations.ensemble.adaboost_emul)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available emulation modules"
    )
    parser.add_argument(
        "--mode",
        choices=["multiprocess", "docker"],
        default="multiprocess",
        help="Emulation mode (default: multiprocess)",
    )

    args = parser.parse_args()

    if args.list:
        print("Available emulation modules:")
        for module in list_emulations():
            print(f"  {module}")
        return

    if not args.module:
        print("Please specify a module to run or use --list to see available modules")
        parser.print_help()
        return

    success = run_emulation(args.module, args.mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
