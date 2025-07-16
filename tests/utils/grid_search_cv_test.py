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

import jax.numpy as jnp
import numpy as np
import pytest
import spu.libspu as libspu
import spu.utils.simulation as spsim
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

from sml.ensemble.adaboost import AdaBoostClassifier
from sml.ensemble.forest import RandomForestClassifier
from sml.gaussian_process._gpc import GaussianProcessClassifier
from sml.linear_model.glm import _GeneralizedLinearRegressor
from sml.linear_model.logistic import LogisticRegression
from sml.linear_model.pla import Perceptron
from sml.linear_model.quantile import QuantileRegressor
from sml.linear_model.ridge import Ridge
from sml.linear_model.sgd_classifier import SGDClassifier
from sml.naive_bayes.gnb import GaussianNB
from sml.neighbors.knn import KNNClassifer
from sml.svm.svm import SVM
from sml.tree.tree import DecisionTreeClassifier
from sml.utils.grid_search_cv import GridSearchCV


@pytest.fixture(scope="module")
def setup_data():
    sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
    random_seed = 42
    n_samples = 60
    n_features = 8
    n_classes_binary = 2
    n_classes_multi = 3
    cv_folds = 2

    X_clf_bin, y_clf_bin = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes_binary,
        random_state=random_seed,
    )
    y_clf_bin = jnp.array(y_clf_bin)
    y_clf_bin_reshaped = y_clf_bin.reshape(-1, 1)

    y_clf_bin_negpos = jnp.where(y_clf_bin == 0, -1, 1)
    y_clf_bin_negpos_reshaped = y_clf_bin_negpos.reshape(-1, 1)

    X_clf_multi, y_clf_multi = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=1,
        n_classes=n_classes_multi,
        n_clusters_per_class=1,
        random_state=random_seed,
    )
    y_clf_multi = jnp.array(y_clf_multi)
    y_clf_multi_reshaped = y_clf_multi.reshape(-1, 1)

    from sml.preprocessing.preprocessing import KBinsDiscretizer

    binner = KBinsDiscretizer(n_bins=2, strategy="uniform")
    X_clf_bin_binary_features = binner.fit_transform(X_clf_bin)

    X_reg, y_reg = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        noise=0.5,
        random_state=random_seed,
    )
    y_reg_reshaped = y_reg.reshape(-1, 1)

    return {
        "sim": sim,
        "random_seed": random_seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes_binary": n_classes_binary,
        "n_classes_multi": n_classes_multi,
        "cv_folds": cv_folds,
        "X_clf_bin": X_clf_bin,
        "y_clf_bin": y_clf_bin,
        "y_clf_bin_reshaped": y_clf_bin_reshaped,
        "y_clf_bin_negpos": y_clf_bin_negpos,
        "y_clf_bin_negpos_reshaped": y_clf_bin_negpos_reshaped,
        "X_clf_multi": X_clf_multi,
        "y_clf_multi": y_clf_multi,
        "y_clf_multi_reshaped": y_clf_multi_reshaped,
        "X_clf_bin_binary_features": X_clf_bin_binary_features,
        "X_reg": X_reg,
        "y_reg": y_reg,
        "y_reg_reshaped": y_reg_reshaped,
    }


def _run_test(
    setup_data,
    model_name,
    estimator,
    param_grid,
    X,
    y,
    scoring,
    task_type,
    refit=False,
    cv_type="iterable",
):
    print(f"\n--- Testing GridSearchCV with {model_name} ---")

    sim = setup_data["sim"]
    cv_folds = setup_data["cv_folds"]

    if cv_type == "iterable":
        if task_type == "classification":
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_splits = [
                (jnp.array(train_idx), jnp.array(test_idx))
                for train_idx, test_idx in skf.split(X, y)
            ]
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_splits = [
                (jnp.array(train_idx), jnp.array(test_idx))
                for train_idx, test_idx in kf.split(X)
            ]
        cv = cv_splits
    elif cv_type == "int":
        cv = cv_folds
    else:
        raise ValueError("cv_type must be 'iterable' or 'int'")

    def run_grid_search_spu(X_spu, y_spu):
        grid_search = GridSearchCV(
            estimator=copy.deepcopy(estimator),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=refit,
            task_type=task_type,
        )
        grid_search.fit(X_spu, y_spu)
        return grid_search.best_score_, grid_search.best_params_

    spu_best_score, spu_best_param = spsim.sim_jax(sim, run_grid_search_spu)(X, y)
    print(f"SPU Best CV Score ({scoring}): {spu_best_score}")
    print(f"SPU Best Params: {spu_best_param}")

    grid_search_plain = GridSearchCV(
        estimator=copy.deepcopy(estimator),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=refit,
        task_type=task_type,
    )
    grid_search_plain.fit(X, y)
    plain_best_score = grid_search_plain.best_score_
    print(f"Plaintext Best CV Score ({scoring}): {plain_best_score}")
    print(f"Plaintext Best Params: {grid_search_plain.best_params_}")

    np.testing.assert_allclose(spu_best_score, plain_best_score, rtol=1e-2, atol=1e-2)
    print(f"--- {model_name} Test Passed ---")

    if refit:
        X_test = X[:10]  # Use a subset for testing
        y_test = y[:10]
        spu_pred = grid_search_plain.predict(X_test)
        spu_score = grid_search_plain.score(X_test, y_test)
        print(f"SPU Prediction: {spu_pred}")
        print(f"SPU Score: {spu_score}")


def test_gridsearch_logistic(setup_data):
    estimator = LogisticRegression(epochs=3, batch_size=16, class_labels=[0, 1])
    param_grid = {"learning_rate": [0.01, 0.1, 0.05], "C": [1.0, 2.0, 5.0]}
    _run_test(
        setup_data,
        "LogisticRegression with cv",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin_reshaped"],
        "accuracy",
        "classification",
        refit=True,
    )


def test_gridsearch_knn(setup_data):
    estimator = KNNClassifer(n_classes=setup_data["n_classes_binary"])
    param_grid = {"n_neighbors": [2, 3, 4, 5]}
    _run_test(
        setup_data,
        "KNNClassifier with cv as iterable",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin"],
        "accuracy",
        "classification",
        refit=True,
    )


def test_gridsearch_gnb(setup_data):
    classes = jnp.unique(setup_data["y_clf_bin"])
    estimator = GaussianNB(classes_=classes, var_smoothing=1e-7)
    param_grid = {"var_smoothing": [1e-6, 2e-6, 1e-5]}
    _run_test(
        setup_data,
        "GaussianNB with cv as int",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin"],
        "accuracy",
        "classification",
        refit=True,
        cv_type="int",
    )


def test_gridsearch_perceptron(setup_data):
    estimator = Perceptron(max_iter=10)
    param_grid = {"alpha": [0.0001, 0.001], "eta0": [0.01, 0.1, 1.0]}
    _run_test(
        setup_data,
        "Perceptron with cv as iterable",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin_negpos_reshaped"],
        "accuracy",
        "classification",
        refit=True,
    )


def test_gridsearch_svm(setup_data):
    estimator = SVM(max_iter=10, C=1.0)
    param_grid = {"C": [0.5, 1.0, 5.0]}
    _run_test(
        setup_data,
        "SVM with cv as int",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin_negpos"],
        "accuracy",
        "classification",
        refit=True,
        cv_type="int",
    )


@pytest.mark.skip("GPC is often slow to settings")
def test_gridsearch_gpc(setup_data):
    estimator = GaussianProcessClassifier(
        max_iter_predict=5, n_classes_=setup_data["n_classes_binary"]
    )
    param_grid = {"max_iter_predict": [1, 3, 5]}
    _run_test(
        setup_data,
        "GaussianProcessClassifier with cv as iterable",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin"],
        "accuracy",
        "classification",
        refit=True,
    )


@pytest.mark.skip("SGDClassifier needs predict() added first")
def test_gridsearch_sgdclassifier(setup_data):
    estimator = SGDClassifier(
        epochs=3,
        learning_rate=0.1,
        batch_size=16,
        reg_type="logistic",
        penalty="l2",
    )
    param_grid = {"learning_rate": [0.1, 0.05], "l2_norm": [0.01, 0.1]}
    _run_test(
        setup_data,
        "SGDClassifier with cv as int",
        estimator,
        param_grid,
        setup_data["X_clf_bin"],
        setup_data["y_clf_bin_reshaped"],
        "accuracy",
        "classification",
        refit=True,
        cv_type="int",
    )


@pytest.mark.skip(
    "FIXME: Test it when we support revealing the best model from SPU during the program."
)
def test_gridsearch_decisiontree(setup_data):
    estimator = DecisionTreeClassifier(
        max_depth=3,
        n_labels=setup_data["n_classes_binary"],
        criterion="gini",
        splitter="best",
    )
    param_grid = {"max_depth": [2, 3, 4]}
    _run_test(
        setup_data,
        "DecisionTreeClassifier with cv as iterable",
        estimator,
        param_grid,
        setup_data["X_clf_bin_binary_features"],
        setup_data["y_clf_bin"],
        "accuracy",
        "classification",
        refit=True,
    )


@pytest.mark.skip(
    "FIXME: Test it when we support revealing the best model from SPU during the program."
)
def test_gridsearch_randomforest(setup_data):
    estimator = RandomForestClassifier(
        n_estimators=3,
        max_depth=3,
        n_labels=setup_data["n_classes_binary"],
        criterion="gini",
        splitter="best",
        max_features=0.5,
        bootstrap=False,
        max_samples=None,
    )
    param_grid = {"max_depth": [2, 3], "n_estimators": [2, 4]}
    _run_test(
        setup_data,
        "RandomForestClassifier with cv",
        estimator,
        param_grid,
        setup_data["X_clf_bin_binary_features"],
        setup_data["y_clf_bin"],
        "accuracy",
        "classification",
        refit=True,
    )


@pytest.mark.skip(
    "FIXME: Test it when we support revealing the best model from SPU during the program."
)
def test_gridsearch_adaboost(setup_data):
    base_estimator = DecisionTreeClassifier(
        max_depth=1,
        n_labels=setup_data["n_classes_binary"],
        criterion="gini",
        splitter="best",
    )
    estimator = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=3,
        learning_rate=1.0,
        algorithm="discrete",
    )
    param_grid = {"n_estimators": [2, 4]}
    _run_test(
        setup_data,
        "AdaBoostClassifier with cv as iterable",
        estimator,
        param_grid,
        setup_data["X_clf_bin_binary_features"],
        setup_data["y_clf_bin"],
        "accuracy",
        "classification",
        refit=True,
    )


def test_gridsearch_ridge(setup_data):
    estimator = Ridge(solver="cholesky")
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    _run_test(
        setup_data,
        "Ridge with cv as int",
        estimator,
        param_grid,
        setup_data["X_reg"],
        setup_data["y_reg_reshaped"],
        "r2",
        "regression",
        refit=True,
        cv_type="int",
    )


def test_gridsearch_glm(setup_data):
    estimator = _GeneralizedLinearRegressor(max_iter=10)
    param_grid = {"alpha": [0.0, 0.1, 0.2]}
    _run_test(
        setup_data,
        "GeneralizedLinearRegressor with cv as iterable",
        estimator,
        param_grid,
        setup_data["X_reg"],
        setup_data["y_reg"],
        "neg_mean_squared_error",
        "regression",
        refit=True,
    )


@pytest.mark.skip(
    "QuantileRegressor requires simplex solver, may be slow/complex for basic test"
)
def test_gridsearch_quantile(setup_data):
    estimator = QuantileRegressor(max_iter=20, lr=0.05)
    param_grid = {"quantile": [0.25, 0.5, 0.75], "alpha": [0.1, 0.5]}
    _run_test(
        setup_data,
        "QuantileRegressor with cv as int",
        estimator,
        param_grid,
        setup_data["X_reg"],
        setup_data["y_reg"],
        "r2",
        "regression",
        refit=True,
        cv_type="int",
    )
