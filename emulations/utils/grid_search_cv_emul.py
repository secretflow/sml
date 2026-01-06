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

import copy
import time

import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

import emulations.utils.emulation as emulation
from sml.linear_model.pla import Perceptron
from sml.linear_model.ridge import Ridge
from sml.naive_bayes.gnb import GaussianNB
from sml.neighbors.knn import KNNClassifer
from sml.svm.svm import SVM
from sml.utils.grid_search_cv import GridSearchCV


def _run_gridsearch_test(
    emulator, model_name, estimator, param_grid, X, y, scoring, task_type, cv_splits
):
    print(f"\n--- Testing GridSearchCV with {model_name} ---")
    grid_search_plain = GridSearchCV(
        estimator=copy.deepcopy(estimator),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_splits,
        refit=False,
        task_type=task_type,
    )
    start_time_plain = time.time()
    grid_search_plain.fit(X, y)
    plain_time = time.time() - start_time_plain
    plain_best_score = grid_search_plain.best_score_
    plain_best_params = grid_search_plain.best_params_
    print(f"Plaintext Best CV Score ({scoring}): {plain_best_score}")
    print(f"Plaintext Best Params: {plain_best_params}")
    print(f"Plaintext Execution Time: {plain_time:.2f}s")

    def run_grid_search_emulated(X_emul, y_emul):
        grid_search = GridSearchCV(
            estimator=copy.deepcopy(estimator),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_splits,
            refit=False,
            task_type=task_type,
        )
        grid_search.fit(X_emul, y_emul)
        return grid_search.best_score_, grid_search.best_params_

    X_sealed = emulator.seal(X)
    y_sealed = emulator.seal(y)
    start_time_emul = time.time()
    emul_best_score, emul_best_params = emulator.run(run_grid_search_emulated)(
        X_sealed, y_sealed
    )
    emul_time = time.time() - start_time_emul
    print(f"Emulated Best CV Score ({scoring}): {emul_best_score}")
    print(f"Emulated Best Params: {emul_best_params}")
    print(f"Emulated Execution Time: {emul_time:.2f}s")
    np.testing.assert_allclose(emul_best_score, plain_best_score, rtol=1e-2, atol=1e-2)
    print(f"--- {model_name} Emulation Test Passed ---")


def emul_comprehensive_gridsearch(emulator: emulation.Emulator):
    print("Starting Comprehensive GridSearchCV Emulation.")
    random_seed = 42
    np.random.seed(random_seed)
    n_samples = 60
    n_features = 8
    n_classes_binary = 2
    n_classes_multi = 3
    cv_folds = 2
    X_clf_bin, y_clf_bin_np = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes_binary,
        random_state=random_seed,
    )
    X_clf_bin = jnp.array(X_clf_bin)
    y_clf_bin = jnp.array(y_clf_bin_np)
    y_clf_bin_negpos = jnp.where(y_clf_bin == 0, -1, 1)
    y_clf_bin_negpos_reshaped = y_clf_bin_negpos.reshape(-1, 1)
    X_clf_multi, y_clf_multi_np = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=1,
        n_classes=n_classes_multi,
        n_clusters_per_class=1,
        random_state=random_seed,
    )
    X_clf_multi = jnp.array(X_clf_multi)
    X_reg, y_reg_np = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        noise=0.5,
        random_state=random_seed,
    )
    X_reg = jnp.array(X_reg)
    y_reg = jnp.array(y_reg_np)
    y_reg_reshaped = y_reg.reshape(-1, 1)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_splits_clf_bin = [
        (jnp.array(train_idx), jnp.array(test_idx))
        for train_idx, test_idx in skf.split(X_clf_bin, y_clf_bin_np)
    ]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_splits_reg = [
        (jnp.array(train_idx), jnp.array(test_idx))
        for train_idx, test_idx in kf.split(X_reg)
    ]

    estimator = KNNClassifer(n_classes=n_classes_binary)
    param_grid = {"n_neighbors": [2, 3, 4, 5]}
    _run_gridsearch_test(
        emulator,
        "KNNClassifier",
        estimator,
        param_grid,
        X_clf_bin,
        y_clf_bin,
        "accuracy",
        "classification",
        cv_splits_clf_bin,
    )
    classes = jnp.unique(y_clf_bin)
    estimator = GaussianNB(classes_=classes, var_smoothing=1e-7)
    param_grid = {"var_smoothing": [1e-6, 2e-6, 1e-5]}
    _run_gridsearch_test(
        emulator,
        "GaussianNB",
        estimator,
        param_grid,
        X_clf_bin,
        y_clf_bin,
        "accuracy",
        "classification",
        cv_splits_clf_bin,
    )
    estimator = Perceptron(max_iter=10, eta0=0.1)
    param_grid = {"alpha": [0.0001, 0.001]}
    _run_gridsearch_test(
        emulator,
        "Perceptron",
        estimator,
        param_grid,
        X_clf_bin,
        y_clf_bin_negpos_reshaped,
        "accuracy",
        "classification",
        cv_splits_clf_bin,
    )
    estimator = SVM(max_iter=10, C=1.0)
    param_grid = {"C": [0.5, 1.0, 5.0]}
    _run_gridsearch_test(
        emulator,
        "SVM",
        estimator,
        param_grid,
        X_clf_bin,
        y_clf_bin_negpos,
        "accuracy",
        "classification",
        cv_splits_clf_bin,
    )
    estimator = Ridge(solver="cholesky")
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    _run_gridsearch_test(
        emulator,
        "Ridge",
        estimator,
        param_grid,
        X_reg,
        y_reg_reshaped,
        "r2",
        "regression",
        cv_splits_reg,
    )
    print("\nComprehensive GridSearchCV Emulation finished successfully.")


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
        emul_comprehensive_gridsearch(emulator)


if __name__ == "__main__":
    main()
