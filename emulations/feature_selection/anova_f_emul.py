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

import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif as f_classif_sklearn

import emulations.utils.emulation as emulation
from sml.feature_selection.anova_f import f_classif


def emul_ANOVA_F(emulator: emulation.Emulator):
    """
    Emulation function for ANOVA F-test with multi-class data.
    """
    print("Start ANOVA F-test multi-class emulation...")

    def load_data():
        print("Loading Iris dataset...")
        x, y = load_iris(return_X_y=True)
        x = x.astype(np.float64)
        y = y.astype(np.int64)
        return x, y

    def proc(x_all_features, y_labels, k):
        fxp_fraction_bits = 26
        epsilon = 2 ** (-fxp_fraction_bits)
        fpmin = 2 ** (-fxp_fraction_bits)
        f_stat, p_val = f_classif(
            x_all_features,
            y_labels,
            k,
            p_value_iter=100,
            epsilon=epsilon,
            fpmin=fpmin,
        )
        return f_stat, p_val

    try:
        x, y = load_data()
        num_classes = len(np.unique(y))
        num_features = x.shape[1]
        print(
            f"Data loaded: {x.shape[0]} samples, {num_features} features, {num_classes} classes."
        )

        print("Calculating sklearn reference...")
        start_time = time.time()
        sklearn_f_stats, sklearn_p_values = f_classif_sklearn(x, y)
        end_time = time.time()
        print("========================================")
        print(f"Running time in SKlearn: {end_time - start_time:.3f}s")
        print("========================================")

        print("Sealing data...")
        X_spu, y_spu = emulator.seal(x, y)
        print("Data sealed.")

        print("Running SPU computation...")
        start_time = time.time()
        f_spu, p_spu = emulator.run(proc, static_argnums=(2,))(
            X_spu, y_spu, num_classes
        )
        end_time = time.time()
        total_spu_time = end_time - start_time
        print(f"SPU computation finished in {total_spu_time:.3f}s.")

        rtol = 1e-1
        atol = 1e-1

        for idx in range(num_features):
            f_ref_val = sklearn_f_stats[idx]
            p_ref_val = sklearn_p_values[idx]
            f_res = f_spu[idx]
            p_res = p_spu[idx]

            print(f"\n--- Feature {idx} ---")
            print(f"SPU Result: F={f_res:.6f}, p={p_res:.6e}")
            print(f"SKL Result: F={f_ref_val:.6f}, p={p_ref_val:.6e}")

            if np.isinf(f_ref_val):
                assert f_res > 1e10, f"F-stat mismatch (Ref: Inf, SPU: {f_res})"
            else:
                assert np.allclose(
                    f_res, f_ref_val, rtol=rtol, atol=atol
                ), f"F-stat mismatch (Ref: {f_ref_val}, SPU: {f_res})"

            if np.isinf(f_ref_val):
                assert p_res < atol, f"P-value mismatch (Ref: 0.0, SPU: {p_res})"
            else:
                try:
                    assert np.allclose(
                        p_res, p_ref_val, rtol=rtol, atol=atol
                    ), f"P-value mismatch (Ref: {p_ref_val}, SPU: {p_res})"
                except AssertionError:
                    if p_ref_val < atol and p_res < atol:
                        print(f"Note: P-values both < {atol:.6e}, accepting.")
                    else:
                        raise

            print(f"Feature {idx} comparison PASSED.")

        print("========================================")
        print(f"Total SPU running time: {total_spu_time:.3f}s")
        print("========================================")
        print("All comparisons passed.")

    except Exception as e:
        print(f"An error occurred during emulation: {e}")
        import traceback

        traceback.print_exc()


def main(
    cluster_config: str = emulation.CLUSTER_ABY3_3PC,
    mode: emulation.Mode = emulation.Mode.MULTIPROCESS,
    bandwidth: int = 300,
    latency: int = 20,
):
    with emulation.start_emulator(
        cluster_config,
        mode,
        bandwidth,
        latency,
    ) as emulator:
        emul_ANOVA_F(emulator)


if __name__ == "__main__":
    main()
