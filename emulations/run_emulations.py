#!/usr/bin/env python3
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

"""
Batch runner script for all SML emulations.

This script automatically discovers and runs all emulation tests in the emulations directory,
providing a comprehensive report of results.
"""

import argparse
import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import emulations.utils.emulation as emulation


@dataclass
class TestResult:
    """Data class to store test execution results."""

    module_path: str
    function_name: str
    success: bool
    duration: float
    error_message: str = ""
    error_type: str = ""


class EmulationRunner:
    """Main runner class for batch emulation execution."""

    def __init__(self, mode: str = "multiprocess", verbose: bool = False):
        """
        Initialize the emulation runner.

        Args:
            mode: Emulation mode ('multiprocess' or 'docker')
            verbose: Whether to show verbose output during execution
        """
        self.mode = (
            emulation.Mode.MULTIPROCESS
            if mode.lower() == "multiprocess"
            else emulation.Mode.DOCKER
        )
        self.verbose = verbose
        self.results: List[TestResult] = []

    def discover_emulation_files(self) -> List[str]:
        """
        Discover all emulation files in the emulations directory.

        Returns:
            List of module paths for emulation files
        """
        emulations_dir = Path(__file__).parent
        emulation_files = []

        for module_dir in emulations_dir.iterdir():
            if module_dir.is_dir() and module_dir.name not in ["__pycache__", ".git"]:
                for emul_file in module_dir.glob("*_emul.py"):
                    module_path = f"emulations.{module_dir.name}.{emul_file.stem}"
                    emulation_files.append(module_path)

        return sorted(emulation_files)

    def discover_emul_functions(self, module) -> List[str]:
        """
        Discover all emul_* functions in a module.

        Args:
            module: The imported module

        Returns:
            List of function names that start with 'emul_'
        """
        functions = []
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isfunction(obj)
                and name.startswith("emul_")
                and not name.startswith("emul_test")
            ):  # Skip test helper functions
                functions.append(name)
        return sorted(functions)

    def run_emulation_function(
        self, module_path: str, function_name: str
    ) -> TestResult:
        """
        Run a specific emulation function and record results.

        Args:
            module_path: Path to the module (e.g., 'emulations.cluster.kmeans_emul')
            function_name: Name of the function to run

        Returns:
            TestResult object with execution details
        """
        start_time = time.time()

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get the function
            func = getattr(module, function_name)

            # Check function signature to determine if it needs mode parameter
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if self.verbose:
                print(f"  Running {function_name}...")

            # Run the function with appropriate parameters
            if "mode" in params:
                func(mode=self.mode)
            else:
                func()

            duration = time.time() - start_time
            return TestResult(
                module_path=module_path,
                function_name=function_name,
                success=True,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            error_message = str(e)

            if self.verbose:
                print(f"  ERROR in {function_name}: {error_type}: {error_message}")
                traceback.print_exc()

            return TestResult(
                module_path=module_path,
                function_name=function_name,
                success=False,
                duration=duration,
                error_message=error_message,
                error_type=error_type,
            )

    def run_all_emulations(self) -> None:
        """
        Run all discovered emulation functions.
        """
        emulation_files = self.discover_emulation_files()

        print(f"🔍 Discovered {len(emulation_files)} emulation files")
        print(f"🚀 Running emulations in {self.mode.name} mode...")
        print("=" * 80)

        for i, module_path in enumerate(emulation_files, 1):
            print(f"\n[{i}/{len(emulation_files)}] Processing {module_path}")

            try:
                module = importlib.import_module(module_path)
                functions = self.discover_emul_functions(module)

                if not functions:
                    print(f"  ⚠️  No emul_ functions found in {module_path}")
                    continue

                print(
                    f"  Found {len(functions)} emul functions: {', '.join(functions)}"
                )

                for function_name in functions:
                    result = self.run_emulation_function(module_path, function_name)
                    self.results.append(result)

                    status = "✅ PASS" if result.success else "❌ FAIL"
                    print(f"  {status} {function_name} ({result.duration:.2f}s)")

            except ImportError as e:
                print(f"  ❌ Failed to import {module_path}: {e}")
                self.results.append(
                    TestResult(
                        module_path=module_path,
                        function_name="<import_error>",
                        success=False,
                        duration=0.0,
                        error_message=str(e),
                        error_type="ImportError",
                    )
                )

    def generate_report(self) -> None:
        """
        Generate and display a comprehensive execution report.
        """
        if not self.results:
            print("\n📊 No results to report.")
            return

        # Calculate statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        total_duration = sum(r.duration for r in self.results)

        # Group results by module
        module_results: Dict[str, List[TestResult]] = {}
        for result in self.results:
            if result.module_path not in module_results:
                module_results[result.module_path] = []
            module_results[result.module_path].append(result)

        # Generate report
        print("\n" + "=" * 80)
        print("📊 EMULATION EXECUTION REPORT")
        print("=" * 80)

        # Summary
        print(f"\n📈 SUMMARY")
        print(f"Total Tests: {total_tests}")
        print(
            f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)"
        )
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {total_duration/total_tests:.2f}s per test")

        # Detailed results by module
        print(f"\n📋 DETAILED RESULTS BY MODULE")
        print("-" * 80)

        for module_path, results in sorted(module_results.items()):
            module_success = sum(1 for r in results if r.success)
            module_total = len(results)
            module_duration = sum(r.duration for r in results)

            status_icon = (
                "✅"
                if module_success == module_total
                else "❌" if module_success == 0 else "⚠️"
            )
            print(f"\n{status_icon} {module_path}")
            print(
                f"   Tests: {module_success}/{module_total} passed, Duration: {module_duration:.2f}s"
            )

            for result in results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"   {status} {result.function_name} ({result.duration:.2f}s)")
                if not result.success:
                    print(
                        f"      Error: {result.error_type}: {result.error_message[:100]}..."
                    )

        # Failed tests summary
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\n❌ FAILED TESTS SUMMARY")
            print("-" * 80)

            for result in failed_results:
                print(f"• {result.module_path}.{result.function_name}")
                print(f"  Error: {result.error_type}: {result.error_message[:150]}...")

        # Top slowest tests
        slowest_tests = sorted(self.results, key=lambda r: r.duration, reverse=True)[
            :10
        ]
        print(f"\n🐌 TOP 10 SLOWEST TESTS")
        print("-" * 80)

        for i, result in enumerate(slowest_tests, 1):
            status = "✅" if result.success else "❌"
            print(
                f"{i:2d}. {status} {result.module_path}.{result.function_name} ({result.duration:.2f}s)"
            )

        print("\n" + "=" * 80)
        print(f"🎉 Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main():
    """Main entry point for the batch emulation runner."""
    parser = argparse.ArgumentParser(
        description="Run all SML emulation tests and generate comprehensive reports"
    )
    parser.add_argument(
        "--mode",
        choices=["multiprocess", "docker"],
        default="multiprocess",
        help="Emulation mode (default: multiprocess)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during execution",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available emulation files without running them",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = EmulationRunner(mode=args.mode, verbose=args.verbose)

    if args.list_only:
        emulation_files = runner.discover_emulation_files()
        print(f"📁 Found {len(emulation_files)} emulation files:")
        for file in emulation_files:
            print(f"  • {file}")
        return

    # Run all emulations
    start_time = time.time()
    print(
        f"🚀 Starting batch emulation run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        runner.run_all_emulations()
        runner.generate_report()

        total_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {total_time:.2f}s")

    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user")
        if runner.results:
            print("Generating partial report...")
            runner.generate_report()
        sys.exit(1)

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
