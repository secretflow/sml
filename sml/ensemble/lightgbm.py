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


import copy

import jax.numpy as jnp

from sml.tree.tree import DecisionTreeClassifier as sml_dtc


class LightGBMClassifier:
    """A LightGBM classifier based on DecisionTreeClassifier.

    Parameters
    ----------
    n_estimators : int
        The number of estimators. Must specify an integer > 0.

    learning_rate : float
        The step size used to update the model weights during training.
        It's a float, must learning_rate > 0.

    max_depth : int
        The maximum depth of each tree. Must specify an integer > 0.

    num_leaves : int
        The maximum number of leaves in each tree.
        Must be greater than 1. Default is 31.

    criterion : {"gini"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity.

    epsilon : float (default=1e-5)
        A small positive value used in calculations to avoid division by zero and other numerical issues.
        Must be greater than 0 and less than 0.1.

    """

    def __init__(
        self,
        n_estimators,
        learning_rate,
        max_depth,
        num_leaves=31,
        criterion="gini",
        epsilon=1e-5,
    ):
        assert n_estimators is not None and n_estimators > 0, (
            "n_estimators should not be None and must > 0."
        )
        assert learning_rate is not None and learning_rate > 0, (
            "learning_rate should not be None and must > 0."
        )
        assert max_depth is not None and max_depth > 0, (
            "max_depth should not be None and must > 0."
        )
        assert num_leaves is not None and num_leaves > 1, (
            "num_leaves should not be None and must > 1."
        )
        assert criterion == "gini", "criteria other than gini is not supported."
        assert epsilon > 0 and epsilon < 0.1, "epsilon must be > 0 and < 0.1."

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.criterion = criterion
        self.epsilon = epsilon

        self.estimators_ = []
        self.estimator_weight_ = jnp.zeros(self.n_estimators, dtype=jnp.float32)
        self.estimator_errors_ = jnp.ones(self.n_estimators, dtype=jnp.float32)
        self.estimator_flags_ = jnp.zeros(self.n_estimators, dtype=jnp.bool_)

    def _num_samples(self, x):
        """Return the number of samples in x."""
        if hasattr(x, "fit"):
            raise TypeError("Expected sequence or array-like, got estimator")
        if (
            not hasattr(x, "__len__")
            and not hasattr(x, "shape")
            and not hasattr(x, "__array__")
        ):
            raise TypeError("Expected sequence or array-like, got %s" % type(x))

        if hasattr(x, "shape"):
            if len(x.shape) == 0:
                raise TypeError(
                    "Singleton array %r cannot be considered a valid collection." % x
                )
            return x.shape[0]
        else:
            return len(x)

    def _check_sample_weight(self, sample_weight, X):
        """
        Description: Validate and process sample weights.

        Parameters:
        - sample_weight: Can be None, a scalar (int or float), or a 1D array-like.
        - X: Input data from which to determine the number of samples.

        Returns:
        - sample_weight: A 1D array of sample weights, one for each sample in X.

        Sample weight scenarios:
        1. None:
           - If sample_weight is None, it will be initialized to an array of ones,
             meaning all samples are equally weighted.
        2. Scalar (int or float):
           - If sample_weight is a scalar, it will be converted to an array where
             each sample's weight is equal to the scalar value.
        3. Array-like:
           - If sample_weight is an array or array-like, it will be converted to a JAX array.
           - The array must be 1D and its length must match the number of samples.
           - If these conditions are not met, an error will be raised.
        """
        n_samples = self._num_samples(X)

        if sample_weight is None:
            sample_weight = jnp.ones(n_samples, dtype=jnp.float32)
        elif isinstance(sample_weight, (jnp.int32, jnp.float32)):
            sample_weight = jnp.full(n_samples, sample_weight, dtype=jnp.float32)
        else:
            sample_weight = jnp.asarray(sample_weight, dtype=jnp.float32)
            if sample_weight.ndim != 1:
                raise ValueError("Sample weight must be 1D array or scalar")

            if sample_weight.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight.shape == {sample_weight.shape}, expected {(n_samples,)}!"
                )

        return sample_weight

    def fit(self, X, y, sample_weight=None):
        sample_weight = self._check_sample_weight(
            sample_weight,
            X,
        )
        sample_weight /= sample_weight.sum()

        self.classes = jnp.unique(y)
        self.n_classes = len(self.classes)

        epsilon = self.epsilon

        for iboost in range(self.n_estimators):
            sample_weight = jnp.clip(sample_weight, a_min=epsilon, a_max=None)

            estimator = sml_dtc(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                n_labels=self.n_classes,
            )

            sample_weight, estimator_weight, estimator_error, flag = (
                self._boost_round(
                    iboost,
                    X,
                    y,
                    sample_weight,
                    estimator,
                )
            )

            self.estimator_weight_ = self.estimator_weight_.at[iboost].set(
                estimator_weight
            )
            self.estimator_errors_ = self.estimator_errors_.at[iboost].set(
                estimator_error
            )
            self.estimator_flags_ = self.estimator_flags_.at[iboost].set(flag)

            sample_weight_sum = jnp.sum(sample_weight)
            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum

        return self

    def _boost_round(self, iboost, X, y, sample_weight, estimator):
        """Implement a single boosting round using gradient boosting approach."""
        self.estimators_.append(estimator)

        n_classes = self.n_classes
        epsilon = self.epsilon

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        incorrect = y_predict != y
        estimator_error = jnp.mean(
            jnp.average(incorrect, weights=sample_weight, axis=0)
        )

        # Check if error is too small
        is_small_error = estimator_error <= epsilon

        def true_fun(sample_weight):
            return sample_weight, 1.0, 0.0, jnp.array(False, dtype=jnp.bool_)

        def false_fun(sample_weight, estimator_error, incorrect, n_classes):
            flag = estimator_error < 1.0 - (1.0 / n_classes)

            estimator_weight = self.learning_rate * (
                jnp.log((1.0 - estimator_error) / estimator_error)
                + jnp.log(n_classes - 1.0)
            )
            sample_weight_updated = sample_weight * jnp.exp(
                estimator_weight * incorrect
            )

            sample_weight = jnp.where(flag, sample_weight_updated, sample_weight)
            estimator_weight = jnp.where(flag, estimator_weight, 0.0)

            return sample_weight, estimator_weight, estimator_error, flag

        sample_weight_true, estimator_weight_true, estimator_error_true, flag_true = (
            true_fun(sample_weight)
        )
        (
            sample_weight_false,
            estimator_weight_false,
            estimator_error_false,
            flag_false,
        ) = false_fun(sample_weight, estimator_error, incorrect, n_classes)

        sample_weight = jnp.where(
            is_small_error, sample_weight_true, sample_weight_false
        )
        estimator_weight = jnp.where(
            is_small_error, estimator_weight_true, estimator_weight_false
        )
        estimator_error = jnp.where(
            is_small_error, estimator_error_true, estimator_error_false
        )
        flag = jnp.where(is_small_error, flag_true, flag_false)

        return sample_weight, estimator_weight, estimator_error, flag

    def predict(self, X):
        pred = self.decision_function(X)

        if self.n_classes == 2:
            return self.classes.take(pred > 0, axis=0)

        return self.classes.take(jnp.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        n_classes = self.n_classes
        classes = self.classes[:, jnp.newaxis]

        pred = sum(
            jnp.where(
                (estimator.predict(X) == classes).T,
                w,
                -1 / (n_classes - 1) * w,
            )
            * flag
            for estimator, w, flag in zip(
                self.estimators_,
                self.estimator_weight_,
                self.estimator_flags_,
                strict=True,
            )
        )

        weights_flags = self.estimator_weight_ * self.estimator_flags_
        pred /= jnp.sum(weights_flags)

        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
