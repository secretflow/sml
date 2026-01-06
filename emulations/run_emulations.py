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
import multiprocessing
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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

    def __init__(
        self,
        mode: str = "multiprocess",
        cluster_config: str | None = None,
        bandwidth: int | None = None,
        latency: int | None = None,
        slowest_tests_count: int = 5,
        target_module: str | None = None,
        verbose: bool = False,
        timeout: int = 300,
    ):
        """
        Initialize the emulation runner.

        Args:
            mode: Emulation mode ('multiprocess' or 'docker')
            cluster_config: Cluster configuration file path
            bandwidth: Network bandwidth limit in Mbps
            latency: Network latency in ms
            slowest_tests_count: Number of slowest tests to show in report
            target_module: Specific module to run (e.g., 'emulations.cluster.kmeans_emul')
            verbose: Whether to show verbose output during execution
            timeout: Timeout for each emulation test in seconds (0 means no timeout)
        """
        # Check for docker mode and raise error
        if mode.lower() == "docker":
            raise ValueError(
                "Docker mode is not yet implemented. Please use 'multiprocess' mode."
            )

        self.mode = (
            emulation.Mode.MULTIPROCESS
            if mode.lower() == "multiprocess"
            else emulation.Mode.DOCKER
        )
        self.cluster_config = cluster_config
        self.bandwidth = bandwidth
        self.latency = latency
        self.slowest_tests_count = slowest_tests_count
        self.target_module = target_module
        self.verbose = verbose
        self.timeout = timeout
        self.results: list[TestResult] = []

    @staticmethod
    def _format_exception(exc: Exception) -> dict[str, str]:
        """Format exception details with line info."""

        tb_exc = traceback.TracebackException.from_exception(exc)
        if tb_exc.stack:
            last_frame = tb_exc.stack[-1]
            location = f"{last_frame.filename}:{last_frame.lineno}"
        else:
            location = "<unknown>"

        return {
            "message": f"{exc} @ {location}",
            "traceback": "".join(tb_exc.format()),
            "location": location,
            "type": type(exc).__name__,
        }

    def discover_emulation_files(self) -> list[str]:
        """
        Discover all emulation files in the emulations directory (including nested subdirectories).

        Returns:
            List of module paths for emulation files
        """
        emulations_dir = Path(__file__).parent
        emulation_files = []

        # Use rglob to recursively find all *_emul.py files
        for emul_file in emulations_dir.rglob("*_emul.py"):
            # Skip files in __pycache__ or .git directories
            if "__pycache__" in emul_file.parts or ".git" in emul_file.parts:
                continue

            # Calculate relative path from emulations directory
            relative_path = emul_file.relative_to(emulations_dir)

            # Convert path to module format: preprocessing/encoding/woe_emul.py -> preprocessing.encoding.woe_emul
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            module_path = f"emulations.{'.'.join(module_parts)}"

            emulation_files.append(module_path)

        return sorted(emulation_files)

    def discover_main_function(self, module) -> bool:
        """
        Check if module has a main function.

        Args:
            module: The imported module

        Returns:
            True if main function exists, False otherwise
        """
        return hasattr(module, "main") and callable(module.main)

    def _run_main_in_process(
        self, module_path: str, result_queue: multiprocessing.Queue
    ) -> None:
        """
        Helper function to run main function in a separate process.

        Args:
            module_path: Path to the module
            result_queue: Queue to store the result
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Check if main function exists
            if not self.discover_main_function(module):
                result_queue.put(
                    {
                        "success": False,
                        "error_message": "No main function found",
                        "error_type": "AttributeError",
                    }
                )
                return

            # Get the main function
            main_func = module.main

            # Prepare arguments for main function
            main_kwargs = {}

            # Check function signature to see what parameters it accepts
            sig = inspect.signature(main_func)
            params = sig.parameters

            # Add arguments based on what the main function accepts and what we have
            if "cluster_config" in params and self.cluster_config is not None:
                main_kwargs["cluster_config"] = self.cluster_config
            if "mode" in params:
                main_kwargs["mode"] = self.mode
            if "bandwidth" in params and self.bandwidth is not None:
                main_kwargs["bandwidth"] = self.bandwidth
            if "latency" in params and self.latency is not None:
                main_kwargs["latency"] = self.latency

            # Run the main function
            if main_kwargs:
                main_func(**main_kwargs)
            else:
                # Call with no arguments if no matching parameters or no override values
                main_func()

            # If we get here, the function completed successfully
            result_queue.put({"success": True})

        except Exception as e:
            formatted = self._format_exception(e)
            result_queue.put(
                {
                    "success": False,
                    "error_message": formatted["message"],
                    "error_type": formatted["type"],
                    "traceback": formatted["traceback"],
                    "location": formatted["location"],
                }
            )

    def run_main_function(self, module_path: str) -> TestResult:
        """
        Run the main function from a specific emulation module in a separate process.

        Args:
            module_path: Path to the module (e.g., 'emulations.cluster.kmeans_emul')

        Returns:
            TestResult object with execution details
        """
        start_time = time.time()

        if self.verbose:
            print(f"  Running main function in separate process...")

        # Create a queue for inter-process communication
        # Note: we use a separate process to run to avoid the grpc termination issue.
        # However, we can not exactly use multiple processes, because the ports will be occupied.
        result_queue = multiprocessing.Queue()

        # Create and start the process
        process = multiprocessing.Process(
            target=self._run_main_in_process, args=(module_path, result_queue)
        )

        try:
            process.start()

            # Wait for the process to complete with a timeout
            timeout_value = None if self.timeout == 0 else self.timeout
            process.join(timeout=timeout_value)

            duration = time.time() - start_time

            # Check if process is still alive (timeout occurred)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)  # Give it 5 seconds to terminate gracefully
                if process.is_alive():
                    process.kill()  # Force kill if still alive

                timeout_msg = (
                    "Process timeout (no timeout limit set)"
                    if self.timeout == 0
                    else f"Process timeout (exceeded {self.timeout} seconds)"
                )
                return TestResult(
                    module_path=module_path,
                    function_name="main",
                    success=False,
                    duration=duration,
                    error_message=timeout_msg,
                    error_type="TimeoutError",
                )

            # Check process exit code
            if process.exitcode != 0 and result_queue.empty():
                return TestResult(
                    module_path=module_path,
                    function_name="main",
                    success=False,
                    duration=duration,
                    error_message=f"Process exited with code {process.exitcode}",
                    error_type="ProcessError",
                )

            # Get result from queue
            if not result_queue.empty():
                try:
                    result = result_queue.get_nowait()

                    if result["success"]:
                        return TestResult(
                            module_path=module_path,
                            function_name="main",
                            success=True,
                            duration=duration,
                        )
                    else:
                        if self.verbose and "traceback" in result:
                            print(
                                f"  ERROR in main: {result['error_type']}: {result['error_message']}"
                            )
                            print(result["traceback"])

                        return TestResult(
                            module_path=module_path,
                            function_name="main",
                            success=False,
                            duration=duration,
                            error_message=result["error_message"],
                            error_type=result["error_type"],
                        )
                except queue.Empty:
                    pass

            # If we get here, something went wrong
            return TestResult(
                module_path=module_path,
                function_name="main",
                success=False,
                duration=duration,
                error_message="No result received from process",
                error_type="ProcessError",
            )

        except Exception as e:
            duration = time.time() - start_time
            formatted = self._format_exception(e)

            if self.verbose:
                print(
                    f"  ERROR in process management: {formatted['type']}: {formatted['message']}"
                )
                traceback.print_exc()

            # Clean up process if still running
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

            return TestResult(
                module_path=module_path,
                function_name="main",
                success=False,
                duration=duration,
                error_message=formatted["message"],
                error_type=formatted["type"],
            )

    def run_all_emulations(self) -> None:
        """
        Run all discovered emulation main functions or a specific target module.
        """
        emulation_files = self.discover_emulation_files()

        # Filter to specific module if target_module is specified
        if self.target_module:
            if self.target_module in emulation_files:
                emulation_files = [self.target_module]
                print(f"🎯 Running specific module: {self.target_module}")
            else:
                available_modules = "\n  • ".join([""] + emulation_files)
                raise ValueError(
                    f"Module '{self.target_module}' not found. Available modules:{available_modules}"
                )
        else:
            print(f"🔍 Discovered {len(emulation_files)} emulation files")

        print(f"🚀 Running emulations in {self.mode.name} mode...")

        # Print configuration info
        config_info = []
        if self.cluster_config:
            config_info.append(f"cluster_config={self.cluster_config}")
        if self.bandwidth:
            config_info.append(f"bandwidth={self.bandwidth}")
        if self.latency:
            config_info.append(f"latency={self.latency}")
        if self.target_module:
            config_info.append(f"target_module={self.target_module}")

        if config_info:
            print(f"📋 Configuration: {', '.join(config_info)}")

        print("=" * 80)

        for i, module_path in enumerate(emulation_files, 1):
            print(f"\n[{i}/{len(emulation_files)}] Processing {module_path}")

            try:
                module = importlib.import_module(module_path)

                if not self.discover_main_function(module):
                    print(f"  ⚠️  No main function found in {module_path}")
                    self.results.append(
                        TestResult(
                            module_path=module_path,
                            function_name="main",
                            success=False,
                            duration=0.0,
                            error_message="No main function found",
                            error_type="AttributeError",
                        )
                    )
                    continue

                print(f"  Found main function")

                result = self.run_main_function(module_path)
                self.results.append(result)

                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"  {status} main ({result.duration:.2f}s)")

            except ImportError as e:
                formatted = self._format_exception(e)
                print(f"  ❌ Failed to import {module_path}: {formatted['message']}")
                self.results.append(
                    TestResult(
                        module_path=module_path,
                        function_name="main",
                        success=False,
                        duration=0.0,
                        error_message=formatted["message"],
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
        module_results: dict[str, list[TestResult]] = {}
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
            : self.slowest_tests_count
        ]
        print(f"\n🐌 TOP {self.slowest_tests_count} SLOWEST TESTS")
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
    # Set multiprocessing start method to 'spawn' for better isolation
    if hasattr(multiprocessing, "set_start_method"):
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method already set, ignore
            pass

    usage_examples = """
    Examples:
      # Run with default parameters(all emulations will be run with default configs)
      python emulations/run_emulations.py

      # Run with custom parameters
      python emulations/run_emulations.py --mode=docker --bandwidth=500 --latency=10

      # Specify cluster configuration
      python emulations/run_emulations.py --cluster_config=path/to/config.json

      # List available emulation files only
      python emulations/run_emulations.py --list-only

      # Run a specific module only
      python emulations/run_emulations.py --module=emulations.cluster.kmeans_emul

      # Verbose output with top 10 slowest tests
      python emulations/run_emulations.py --verbose --slowest-tests=10

      # Run with custom timeout (e.g., 600 seconds)
      python emulations/run_emulations.py --timeout=600

      # Run with no timeout limit (useful for long-running tests)
      python emulations/run_emulations.py --timeout=0
    """

    parser = argparse.ArgumentParser(
        description="Run all SML emulation tests and generate comprehensive reports",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["multiprocess", "docker"],
        default="multiprocess",
        help="Emulation mode (default: multiprocess). Note: docker mode is not yet implemented.",
    )
    parser.add_argument(
        "--cluster_config",
        type=str,
        help="Path to cluster configuration file (will override default configs in emul files)",
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        help="Network bandwidth limit in Mbps (for docker mode)",
    )
    parser.add_argument(
        "--latency",
        type=int,
        help="Network latency in milliseconds (for docker mode)",
    )
    parser.add_argument(
        "--slowest-tests",
        type=int,
        default=5,
        help="Number of slowest tests to show in the report (default: 5)",
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Run a specific module only (e.g., 'emulations.cluster.kmeans_emul')",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each emulation test in seconds (default: 300). Use 0 for no timeout limit.",
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
    try:
        runner = EmulationRunner(
            mode=args.mode,
            cluster_config=args.cluster_config,
            bandwidth=args.bandwidth,
            latency=args.latency,
            slowest_tests_count=getattr(args, "slowest_tests", 5),
            target_module=args.module,
            verbose=args.verbose,
            timeout=args.timeout,
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    if args.list_only:
        emulation_files = runner.discover_emulation_files()

        # Filter to specific module if target_module is specified
        if runner.target_module:
            if runner.target_module in emulation_files:
                emulation_files = [runner.target_module]
                print(f"🎯 Specified module: {runner.target_module}")
            else:
                available_modules = "\n  • ".join([""] + emulation_files)
                print(
                    f"❌ Module '{runner.target_module}' not found. Available modules:{available_modules}"
                )
                return
        else:
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
