# Copyright 2023 Ant Group Co., Ltd.
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

import os
import sys

import numpy as np
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import emulations.utils.emulation as emulation

from sml.decomposition.nmf import NMF


def evaluate_nmf_quality(X, W, H, reconstruction_err=None):
    """
    Evaluate NMF factorization quality metrics and correlation metrics

    Args:
        X: Original data matrix
        W: Learned basis matrix (n_samples x n_components)
        H: Learned coefficient matrix (n_components x n_features)
        reconstruction_err: Optional reconstruction error

    Returns:
        dict: Dictionary containing factorization quality metrics and correlation metrics
    """
    X_reconstructed = np.dot(W, H)
    metrics = {}

    # 1. Factorization Quality Metrics
    print("\n=== Factorization Quality Metrics ===")

    # Condition Number - evaluate numerical stability
    try:
        cond_W = np.linalg.cond(W)
        cond_H = np.linalg.cond(H)
        metrics["condition_number_W"] = cond_W
        metrics["condition_number_H"] = cond_H
        print(f"Condition Number W: {cond_W:.4f}")
        print(f"Condition Number H: {cond_H:.4f}")

        # Assess numerical stability
        if cond_W < 100 and cond_H < 100:
            print("✓ Good numerical stability (condition numbers < 100)")
        elif cond_W < 1000 and cond_H < 1000:
            print("⚠ Moderate numerical stability (condition numbers < 1000)")
        else:
            print("✗ Poor numerical stability (condition numbers too large)")
    except np.linalg.LinAlgError:
        metrics["condition_number_W"] = np.inf
        metrics["condition_number_H"] = np.inf
        print("✗ Cannot compute condition numbers")

    # Sparsity - NMF typically pursues sparse representation
    threshold = 1e-10
    sparsity_W = np.sum(np.abs(W) < threshold) / W.size
    sparsity_H = np.sum(np.abs(H) < threshold) / H.size
    metrics["sparsity_W"] = sparsity_W
    metrics["sparsity_H"] = sparsity_H
    print(f"Sparsity W: {sparsity_W:.4f}")
    print(f"Sparsity H: {sparsity_H:.4f}")

    # Non-negativity constraint check
    non_negative_W = np.all(W >= -1e-10)
    non_negative_H = np.all(H >= -1e-10)
    metrics["non_negative_W"] = non_negative_W
    metrics["non_negative_H"] = non_negative_H
    print(f"Non-negativity constraint W: {'✓' if non_negative_W else '✗'}")
    print(f"Non-negativity constraint H: {'✓' if non_negative_H else '✗'}")

    # 2. Correlation Metrics
    print("\n=== Correlation Metrics ===")

    # Flatten matrices for correlation calculation
    X_flat = X.flatten()
    X_recon_flat = X_reconstructed.flatten()

    # R² Score - explained variance ratio
    r2 = r2_score(X_flat, X_recon_flat)
    metrics["r2_score"] = r2
    print(f"R² Score: {r2:.4f}")

    if r2 > 0.9:
        print("✓ Excellent reconstruction quality (R² > 0.9)")
    elif r2 > 0.8:
        print("✓ Good reconstruction quality (R² > 0.8)")
    elif r2 > 0.6:
        print("⚠ Moderate reconstruction quality (R² > 0.6)")
    else:
        print("✗ Poor reconstruction quality (R² < 0.6)")

    # Pearson correlation coefficient - linear correlation
    try:
        pearson_corr, pearson_p = pearsonr(X_flat, X_recon_flat)
        metrics["pearson_correlation"] = pearson_corr
        metrics["pearson_p_value"] = pearson_p
        print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    except:
        metrics["pearson_correlation"] = np.nan
        metrics["pearson_p_value"] = np.nan
        print("✗ Cannot compute Pearson correlation")

    # Spearman correlation coefficient - monotonic correlation
    try:
        spearman_corr, spearman_p = spearmanr(X_flat, X_recon_flat)
        metrics["spearman_correlation"] = spearman_corr
        metrics["spearman_p_value"] = spearman_p
        print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    except:
        metrics["spearman_correlation"] = np.nan
        metrics["spearman_p_value"] = np.nan
        print("✗ Cannot compute Spearman correlation")

    # 3. Reconstruction Error Metrics
    print("\n=== Reconstruction Error Metrics ===")
    frobenius_norm = np.linalg.norm(X - X_reconstructed, "fro")
    normalized_frobenius = frobenius_norm / np.linalg.norm(X, "fro")
    metrics["frobenius_norm"] = frobenius_norm
    metrics["normalized_frobenius"] = normalized_frobenius

    print(f"Frobenius norm: {frobenius_norm:.4f}")
    print(f"Normalized Frobenius norm: {normalized_frobenius:.4f}")

    if reconstruction_err is not None:
        print(f"Algorithm internal reconstruction error: {reconstruction_err:.4f}")
        metrics["algorithm_reconstruction_error"] = reconstruction_err

    return metrics


def emul_nmf(emulator: emulation.Emulator):
    """
    Improved NMF test function using quality metrics evaluation instead of hard comparison
    """

    random_seed = 0
    np.random.seed(random_seed)
    test_data = np.random.randint(1, 100, (100, 10)) * 1.0
    n_samples, n_features = test_data.shape
    n_components = 5
    random_state = np.random.RandomState(random_seed)
    A = random_state.standard_normal(size=(n_components, n_features))
    B = random_state.standard_normal(size=(n_samples, n_components))
    l1_ratio = 0.1
    alpha_W = 0.01

    def nmf_unified():
        def proc1(X):
            model = NMF(
                n_components=n_components,
                l1_ratio=l1_ratio,
                alpha_W=alpha_W,
                random_matrixA=A,
                random_matrixB=B,
            )

            W = model.fit_transform(X)
            H = model.components_
            X_reconstructed = model.inverse_transform(W)
            err = model.reconstruction_err_
            return W, H, X_reconstructed, err

        X_spu = emulator.seal(test_data)

        # Run the emulation
        W, H, X_reconstructed, err = emulator.run(proc1)(X_spu)

        print("\n" + "=" * 60)
        print("SML NMF Unified fit_transform Test")
        print("=" * 60)

        # Use new evaluation method
        metrics = evaluate_nmf_quality(test_data, W, H, err)

        # Simple sklearn baseline comparison (for reference only)
        print("\n=== sklearn Baseline Comparison ===")
        model_sklearn = SklearnNMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
            l1_ratio=l1_ratio,
            solver="mu",
            alpha_W=alpha_W,
        )
        W_sklearn = model_sklearn.fit_transform(test_data)
        H_sklearn = model_sklearn.components_
        err_sklearn = model_sklearn.reconstruction_err_

        metrics_sklearn = evaluate_nmf_quality(
            test_data, W_sklearn, H_sklearn, err_sklearn
        )

        print(
            f"\nSML vs sklearn R² comparison: {metrics['r2_score']:.4f} vs {metrics_sklearn['r2_score']:.4f}"
        )
        print(
            f"SML vs sklearn reconstruction error comparison: {err:.4f} vs {err_sklearn:.4f}"
        )

        return W, H, X_reconstructed, err, metrics, metrics_sklearn

    def nmf_seperate():
        def proc2(X):
            model = NMF(
                n_components=n_components,
                l1_ratio=l1_ratio,
                alpha_W=alpha_W,
                random_matrixA=A,
                random_matrixB=B,
            )

            model.fit(X)
            W = model.transform(X, transform_iter=40)
            H = model.components_
            X_reconstructed = model.inverse_transform(W)
            return W, H, X_reconstructed

        X_spu = emulator.seal(test_data)

        # Run the emulation_separate
        W_seperate, H_seperate, X_reconstructed_seperate = emulator.run(proc2)(X_spu)

        print("\n" + "=" * 60)
        print("SML NMF Separate fit-transform Test")
        print("=" * 60)

        # Use new evaluation method
        metrics = evaluate_nmf_quality(test_data, W_seperate, H_seperate)

        # Simple sklearn baseline comparison (for reference only)
        print("\n=== sklearn Baseline Comparison ===")
        model_sklearn = SklearnNMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
            l1_ratio=l1_ratio,
            solver="mu",
            alpha_W=alpha_W,
        )
        model_sklearn.fit(test_data)
        W_sklearn_seperate = model_sklearn.transform(test_data)
        H_sklearn_seperate = model_sklearn.components_

        metrics_sklearn = evaluate_nmf_quality(
            test_data, W_sklearn_seperate, H_sklearn_seperate
        )

        print(
            f"\nSML vs sklearn R² comparison: {metrics['r2_score']:.4f} vs {metrics_sklearn['r2_score']:.4f}"
        )

        return (
            W_seperate,
            H_seperate,
            X_reconstructed_seperate,
            metrics,
            metrics_sklearn,
        )

    # Run tests
    _, _, _, _, metrics_unified, metrics_sklearn_unified = nmf_unified()
    _, _, _, metrics_separate, metrics_sklearn_separate = nmf_seperate()

    # Final summary
    print("\n" + "=" * 80)
    print("Final Evaluation Summary")
    print("=" * 80)
    print("SML NMF Test Result:")
    print(f"Unified method R²: {metrics_unified['r2_score']:.4f}")
    print(f"Separate method R²: {metrics_separate['r2_score']:.4f}")
    print(f"Unified method Frobenius norm: {metrics_unified['frobenius_norm']:.4f}")
    print(f"Separate method Frobenius norm: {metrics_separate['frobenius_norm']:.4f}")
    print("Sklearn NMF Test Result:")
    print(f"Unified method R²: {metrics_sklearn_unified['r2_score']:.4f}")
    print(f"Separate method R²: {metrics_sklearn_separate['r2_score']:.4f}")
    print(
        f"Unified method Frobenius norm: {metrics_sklearn_unified['frobenius_norm']:.4f}"
    )
    print(
        f"Separate method Frobenius norm: {metrics_sklearn_separate['frobenius_norm']:.4f}"
    )

    np.testing.assert_allclose(
        metrics_unified["r2_score"],
        metrics_sklearn_unified["r2_score"],
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        metrics_separate["r2_score"],
        metrics_sklearn_separate["r2_score"],
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        metrics_unified["frobenius_norm"],
        metrics_sklearn_unified["frobenius_norm"],
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        metrics_separate["frobenius_norm"],
        metrics_sklearn_separate["frobenius_norm"],
        atol=1e-2,
        rtol=1e-2,
    )

    print("=" * 80)


def main(
    cluster_config: str = "emulations/decomposition/3pc.json",
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
        emul_nmf(emulator)


if __name__ == "__main__":
    main()
