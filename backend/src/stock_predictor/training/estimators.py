import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from stock_predictor.config import PipelineError
from stock_predictor.training.metrics import ALPHAS, CLASSES


def fit_booster(
    train_x, train_y, validation_x, validation_y, candidate, config, alpha=None, rounds=None
):
    params = {
        "objective": "quantile" if alpha is not None else "multiclass",
        "metric": "quantile" if alpha is not None else "multi_logloss",
        "learning_rate": config["learning_rate"],
        "feature_fraction": 0.9,
        "seed": config["seed"],
        "num_threads": config["threads"],
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        **{k: v for k, v in candidate.items() if k != "name"},
    }
    if alpha is None:
        params["num_class"] = 3
    else:
        params["alpha"] = alpha
    train = lgb.Dataset(train_x, label=train_y, free_raw_data=True)
    valid = (
        lgb.Dataset(validation_x, label=validation_y, reference=train)
        if validation_x is not None
        else None
    )
    callbacks = (
        [lgb.early_stopping(config["early_stopping_rounds"], verbose=False)]
        if valid is not None
        else []
    )
    return lgb.train(
        params,
        train,
        num_boost_round=rounds or config["max_rounds"],
        valid_sets=[valid] if valid is not None else None,
        callbacks=callbacks,
    )


def fit_linear(x, y, kind, value):
    estimator = (
        Ridge(alpha=value)
        if kind == "ridge"
        else LogisticRegression(C=value, max_iter=1500, solver="lbfgs")
    )
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    model.fit(x, y)
    if kind == "logistic" and list(model[-1].classes_) != [0, 1, 2]:
        raise PipelineError("Direction training must include all three classes")
    return model


def linear_payload(model, kind, feature_columns) -> dict:
    return {
        "kind": kind,
        "feature_columns": feature_columns,
        "imputation_medians": model[0].statistics_.tolist(),
        "scaler_mean": model[1].mean_.tolist(),
        "scaler_scale": model[1].scale_.tolist(),
        "coefficients": np.asarray(model[-1].coef_).tolist(),
        "intercept": np.asarray(model[-1].intercept_).tolist(),
        "classes": CLASSES if kind == "logistic" else None,
    }


def predict_linear(payload, x):
    values = np.asarray(x, dtype=float)
    values = np.where(np.isnan(values), np.asarray(payload["imputation_medians"]), values)
    standardized = (values - payload["scaler_mean"]) / payload["scaler_scale"]
    coef = np.asarray(payload["coefficients"])
    score = standardized @ (coef.T if coef.ndim > 1 else coef) + payload["intercept"]
    return softmax(score, axis=1) if payload["kind"] == "logistic" else score


def fit_arima_bundle(
    x: pd.DataFrame, index: pd.DataFrame, order: int, config: dict, session_positions: dict
) -> dict:
    rows = index[["symbol", "session_date"]].copy()
    rows["return"] = x.ret_1d.to_numpy()
    rows["session_position"] = rows.session_date.map(session_positions)
    parameters = {}
    for symbol, group in rows.groupby("symbol", sort=True):
        group = group.sort_values("session_date")
        segments = group.session_position.diff().ne(1).cumsum()
        blocks = [
            b for _, b in group.groupby(segments) if len(b) >= config["arima_minimum_sessions"]
        ]
        if not blocks:
            raise PipelineError(f"Insufficient contiguous observations for ARIMA: {symbol}")
        group = blocks[-1]
        # Fit percentage log returns to avoid optimizer precision issues on tiny numbers.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = ARIMA(
                group["return"].to_numpy() * 100,
                order=(order, 0, 0),
                trend="c",
                enforce_stationarity=True,
            ).fit(method_kwargs={"maxiter": 200})
        estimated = dict(zip(fitted.param_names, fitted.params, strict=True))
        if not fitted.mle_retvals.get("converged", True):
            raise PipelineError(f"ARIMA order {order} did not converge for {symbol}")
        mean, phi, variance = (
            estimated["const"] / 100,
            estimated.get("ar.L1", 0),
            estimated["sigma2"] / 10000,
        )
        if not np.isfinite([mean, phi, variance]).all() or variance <= 0 or abs(phi) >= 1:
            raise PipelineError(f"Invalid stationary ARIMA parameters for {symbol}")
        parameters[symbol] = {
            "mean": float(mean),
            "phi": float(phi),
            "innovation_variance": float(variance),
            "training_rows": len(group),
            "first_session": group.session_date.iloc[0],
            "last_session": group.session_date.iloc[-1],
            "warning_types": sorted({type(w.message).__name__ for w in captured}),
        }
    return {
        "kind": "arima_returns",
        "order": [order, 0, 0],
        "horizon_sessions": 5,
        "parameters": parameters,
        "update_policy": "Fixed coefficients; condition on the observed return at each cutoff.",
    }


def predict_arima(payload, x, index) -> np.ndarray:
    predictions = np.empty((len(x), 3))
    for symbol, positions in index.groupby("symbol").indices.items():
        if symbol not in payload["parameters"]:
            raise PipelineError("ARIMA model has no parameters for a requested stock")
        params = payload["parameters"][symbol]
        phi, mean, variance = params["phi"], params["mean"], params["innovation_variance"]
        horizon = payload["horizon_sessions"]
        phi_sum = sum(phi**k for k in range(1, horizon + 1))
        endpoint_mean = horizon * mean + (x.iloc[positions].ret_1d.to_numpy() - mean) * phi_sum
        endpoint_variance = variance * sum(
            sum(phi**k for k in range(j + 1)) ** 2 for j in range(horizon)
        )
        predictions[positions] = (
            endpoint_mean[:, None] + np.sqrt(endpoint_variance) * norm.ppf(ALPHAS)[None, :]
        )
    return predictions
