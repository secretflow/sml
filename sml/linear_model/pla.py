# Copyright 2023 Ant Group Co., Ltd.
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


import math
from enum import Enum

import jax
import jax.numpy as jnp

from sml.utils.utils import sml_reveal


class Penalty(Enum):
    NONE = None
    L1 = "l1"
    L2 = "l2"
    EN = "elasticnet"


class EarlyStoppingMetric(Enum):
    HINGE_LOSS = "hinge_loss"
    WEIGHT = "weight"


class Perceptron:
    """Perceptron.

    Parameters
    ----------

    penalty : {'l2','l1','elasticnet',None(NoneType)}, default=None
        The penalty (aka regularization term) to be used.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term if regularization is
        used.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`.
        `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1.
        Only used if `penalty='elasticnet'`.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

    eta0 : float, default=1
        Constant by which the updates are multiplied.

    batch_size : int, default=-1
        The batch size is a number of samples processed before the model is updated.
        if batch_size = -1, all samples will be processed in an iteration.

    early_stopping : bool, default=True
        Whether to enable early stopping to avoid overtraining.

    early_stopping_threshold : float, default=0.0
        The threshold for early stopping based on weight change.
        If > 0, training stops when the maximum absolute weight change
        is below this threshold. Only used when early_stopping_metric='weight'.
        Default: 0.0 (disabled, rely on hinge_loss patience instead).

    early_stopping_metric : str, default='hinge_loss'
        The metric for early stopping:
        - 'hinge_loss': stop when hinge loss hasn't improved for `patience` epochs.
          Best weights (those achieving lowest loss) are restored.
        - 'weight': stop when max(|w - w_last|) < early_stopping_threshold.

    patience : int, default=10
        Number of epochs with no improvement in hinge loss to wait before
        stopping. Only used when early_stopping_metric='hinge_loss'.
    """

    def __init__(
        self,
        penalty=None,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        eta0=1.0,
        batch_size=-1,
        early_stopping=True,
        early_stopping_threshold: float = 0.0,
        early_stopping_metric: str = "hinge_loss",
        patience=10,
    ):
        # parameter check.
        assert max_iter > 0, f"max_iter should >0"
        assert eta0 > 0, f"eta0 should >0"
        assert alpha > 0, f"alpha should >0"
        assert patience > 0, f"patience should >0"
        assert batch_size > 0 or batch_size == -1, f"batch size should >0 or -1"
        assert early_stopping_threshold >= 0, (
            f"early_stopping_threshold should >=0, got {early_stopping_threshold}"
        )
        assert early_stopping_metric in [e.value for e in EarlyStoppingMetric], (
            f"early_stopping_metric should in {[e.value for e in EarlyStoppingMetric]}, but got {early_stopping_metric}"
        )
        assert penalty in [e.value for e in Penalty], (
            f"penalty should in {[e.value for e in Penalty]}, but got {penalty}"
        )
        if penalty == "elasticnet":
            assert l1_ratio > 0 and l1_ratio < 1, (
                f"l1_ratio should >0 and <1 if use ElasticNet (L1+L2)"
            )
        if early_stopping_metric == "weight":
            assert early_stopping_threshold > 0, (
                f"early_stopping_threshold should >0 when early_stopping_metric='weight'"
            )

        self.max_iter = max_iter
        self.eta0 = eta0
        self.fit_intercept = fit_intercept
        self.penalty = Penalty(penalty)
        self.alpha = alpha
        self.patience = patience
        self.bsize = batch_size
        self.early_stopping = early_stopping
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_metric = EarlyStoppingMetric(early_stopping_metric)

        if self.penalty == Penalty.L1:
            self.l1_ratio = 1
        elif self.penalty == Penalty.L2:
            self.l1_ratio = 0
        elif self.penalty == Penalty.EN:
            self.l1_ratio = l1_ratio
        else:
            self.l1_ratio = None

        self._w = None
        self._b = 0.0

        # Track actual training epochs
        self.actual_epochs_ = 0

    def _sign(self, x):
        """
        The sign function in perceptron.
        if x <= 0, f(x)=-1 else if x > 0, f(x)=1
        """
        x = jnp.where(x <= 0, -1, 1)
        return x

    def _hinge_function(self, x, y, w, b):
        """
        loss = \\frac{1}{n} \\sum_{i=1}^{n}max\\{-y_i(wx_i+b), 0\\}.
        """
        y_hat = jnp.matmul(x, w) + b
        y_hat = y_hat.reshape(y_hat.shape[0], 1)
        return jnp.mean(jnp.maximum(jnp.multiply(-y_hat, y), 0))

    def _update_parameters(self, x_i, y_i, w, b):
        # if y_i * [(w * x_i) + b] <= 0, the point is misclassified used for updating w and b.
        decision = y_i * (jnp.dot(w, x_i) + b) <= 0

        dw = decision * (self.eta0 * y_i * x_i)
        if self.fit_intercept:
            db = decision * self.eta0 * y_i
        else:
            db = jnp.array([0.0])

        # add regularization terms
        if self.penalty != Penalty.NONE:
            l1_reg = jnp.sign(w)
            l2_reg = w
            reg = self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg
            dw += decision * self.alpha * reg

        dwb = jnp.concatenate((dw, db), axis=0)

        return dwb

    def _run_one_epoch(self, x, y, w, b, n_samples):
        """Run one epoch of perceptron training.

        Args:
            x: Feature matrix.
            y: Target values.
            w: Current weights.
            b: Current bias.
            n_samples: Number of samples.

        Returns:
            Updated (w, b).
        """
        total_batch = math.ceil(float(n_samples) / self.bsize)
        for idx in range(total_batch):
            begin = idx * self.bsize
            end = min((idx + 1) * self.bsize, n_samples)
            x_slice = x[begin:end]
            y_slice = y[begin:end]

            update_w_b = jax.vmap(
                self._update_parameters, in_axes=(0, 0, None, None), out_axes=0
            )(x_slice, y_slice, w, b)
            mean_w_b = jnp.mean(jnp.array(update_w_b), axis=0)

            w = w + mean_w_b[:-1]

            if self.fit_intercept:
                b = b + mean_w_b[-1]

        return w, b

    def fit(self, x, y):
        """Fit Perceptron classifier.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values. Only -1 or 1.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        assert len(x.shape) == 2, f"expect x to be 2 dimension array, got {x.shape}"

        n_samples, n_features = x.shape
        if self.bsize == -1 or self.bsize > n_samples:
            self.bsize = n_samples

        w = jnp.zeros(n_features)
        b = jnp.array([0.0])

        if not self.early_stopping:
            # No early stopping: run fixed max_iter epochs
            for iter in range(self.max_iter):
                w, b = self._run_one_epoch(x, y, w, b, n_samples)
            self._w = w
            self._b = b
            self.actual_epochs_ = self.max_iter
            return self

        if self.early_stopping_metric == EarlyStoppingMetric.WEIGHT:
            # Weight-based early stopping using while_loop
            # (similar to logistic.py's approach for SPU compatibility)
            w_last = w
            b_last = b

            def cond_fun(carry):
                _, _, _, _, need_stop = carry
                epoch = carry[2]
                return jnp.logical_and(epoch < self.max_iter, jnp.logical_not(need_stop))

            def body_fun(carry):
                w, b, epoch, _, _ = carry
                w_last = w
                b_last = b
                w, b = self._run_one_epoch(x, y, w, b, n_samples)

                # Check early stopping: max absolute change in weights and bias
                w_delta = jnp.max(jnp.abs(w - w_last))
                b_delta = jnp.max(jnp.abs(b - b_last))
                max_delta = jnp.maximum(w_delta, b_delta)
                stop = max_delta < self.early_stopping_threshold

                # WARNING: Reveal the need_stop to plaintext here
                need_stop = sml_reveal(stop)

                return (w, b, epoch + 1, max_delta, need_stop)

            w, b, final_epoch, _, _ = jax.lax.while_loop(
                cond_fun, body_fun, (w, b, 0, jnp.inf, False)
            )
            self._w = w
            self._b = b
            self.actual_epochs_ = final_epoch

        elif self.early_stopping_metric == EarlyStoppingMetric.HINGE_LOSS:
            # Hinge loss based early stopping with patience
            best_loss = jnp.inf
            best_w = w
            best_b = b
            best_iter = -1
            not_early_stop = True

            for iter in range(self.max_iter):
                w, b = self._run_one_epoch(x, y, w, b, n_samples)

                sumloss = self._hinge_function(x, y, w, b)

                best_loss_flag = sumloss < best_loss
                best_loss = jnp.where(best_loss_flag, sumloss, best_loss)
                best_w = jnp.where(best_loss_flag, w, best_w)
                best_b = jnp.where(best_loss_flag, b, best_b)
                best_iter = jnp.where(best_loss_flag, iter, best_iter)

                not_early_stop *= iter <= best_iter + self.patience

            self._w = best_w
            self._b = best_b
            self.actual_epochs_ = int(best_iter) + 1

        return self

    def predict(self, x):
        """Perceptron estimates.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples,)
            Returns the result of the sample for each class in the model.
        """
        assert self._w is not None, f"the model should be trained before prediction."

        pred_ = jnp.matmul(x, self._w) + self._b
        return self._sign(pred_)