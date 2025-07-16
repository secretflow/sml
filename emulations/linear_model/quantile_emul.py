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

import time

import jax.numpy as jnp
from sklearn.linear_model import QuantileRegressor as SklearnQuantileRegressor

import emulations.utils.emulation as emulation
from sml.linear_model.quantile import QuantileRegressor as SmlQuantileRegressor


def emul_quantile(emulator: emulation.Emulator):
    def proc_wrapper(
        quantile,
        alpha,
        fit_intercept,
        lr,
        max_iter,
    ):
        quantile_custom = SmlQuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            fit_intercept=fit_intercept,
            lr=lr,
            max_iter=max_iter,
        )

        def proc(X, y):
            quantile_custom_fit = quantile_custom.fit(X, y)
            result = quantile_custom_fit.predict(X)
            return result, quantile_custom_fit.coef_, quantile_custom_fit.intercept_

        return proc

    def generate_data():
        from jax import random

        key = random.PRNGKey(42)
        key, subkey = random.split(key)
        X = random.normal(subkey, (50, 2))
        y = 5 * X[:, 0] + 2 * X[:, 1] + random.normal(key, (50,)) * 0.1
        return X, y

    # load mock data
    X, y = generate_data()

    # compare with sklearn
    quantile_sklearn = SklearnQuantileRegressor(
        quantile=0.2, alpha=0.1, fit_intercept=True, solver="highs"
    )
    start = time.time()
    quantile_sklearn_fit = quantile_sklearn.fit(X, y)
    y_pred_plain = quantile_sklearn_fit.predict(X)
    rmse_plain = jnp.sqrt(jnp.mean((y - y_pred_plain) ** 2))
    end = time.time()
    print(f"Running time in SKlearn: {end - start:.2f}s")
    print(quantile_sklearn_fit.coef_)
    print(quantile_sklearn_fit.intercept_)

    # mark these data to be protected in SPU
    X_spu, y_spu = emulator.seal(X, y)

    # run larger max_iter can give higher accuracy, but it will take more time to run
    # Note: here we use a very small max_iter to avoid the timeout issue.
    proc = proc_wrapper(
        quantile=0.2, alpha=0.1, fit_intercept=True, lr=0.01, max_iter=10
    )
    start = time.time()
    result, coef, intercept = emulator.run(proc)(X_spu, y_spu)
    end = time.time()
    rmse_encrpted = jnp.sqrt(jnp.mean((y - result) ** 2))
    print(f"Running time in SPU: {end - start:.2f}s")
    print(coef)
    print(intercept)

    # print RMSE
    print(f"RMSE in SKlearn: {rmse_plain:.2f}")
    print(f"RMSE in SPU: {rmse_encrpted:.2f}")


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
        emul_quantile(emulator)


if __name__ == "__main__":
    main()
