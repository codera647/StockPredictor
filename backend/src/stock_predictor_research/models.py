import copy

import numpy as np
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from stock_predictor.config import PipelineError
from stock_predictor.training.estimators import fit_linear, linear_payload, predict_linear
from stock_predictor.training.metrics import encode_labels, validate_probabilities
from stock_predictor_challengers.models import fit_model

from stock_predictor_research.macro import RESEARCH_FEATURES


def neural_payload(model, imputer, scaler):
    return {
        "kind": "neural",
        "feature_columns": RESEARCH_FEATURES,
        "imputation_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": [c.tolist() for c in model.coefs_],
        "intercepts": [i.tolist() for i in model.intercepts_],
        "activation": "relu",
        "class_order": ["DOWN", "FLAT", "UP"],
    }


def predict_neural(payload, x):
    if list(x) != payload["feature_columns"]:
        raise PipelineError("Neural inference schema differs from its saved feature ordering")
    values = np.asarray(x, dtype=float)
    values = np.where(np.isnan(values), np.asarray(payload["imputation_medians"]), values)
    values = (values - payload["scaler_mean"]) / payload["scaler_scale"]
    for i, (coef, intercept) in enumerate(
        zip(payload["coefficients"], payload["intercepts"], strict=True)
    ):
        values = values @ np.asarray(coef) + np.asarray(intercept)
        if i < len(payload["coefficients"]) - 1:
            values = np.maximum(values, 0)
    return softmax(values, axis=1)


def fit_components(train, validation, config):
    train_y = encode_labels(train["y"].y_direction_5d)
    valid_y = encode_labels(validation["y"].y_direction_5d)
    trials, best_loss, logistic = [], float("inf"), None
    for c in config["logistic_cs"]:
        model = fit_linear(train["X"], train_y, "logistic", c)
        score = float(log_loss(valid_y, model.predict_proba(validation["X"]), labels=[0, 1, 2]))
        trials.append({"model": "macro_logistic", "C": c, "log_loss": score})
        if score < best_loss:
            best_loss, logistic = score, linear_payload(model, "logistic", RESEARCH_FEATURES)
    tree, rounds = fit_model(
        "catboost", train["X"], train_y, validation["X"], valid_y, config["catboost"], config
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(train["X"]))
    valid_x = scaler.transform(imputer.transform(validation["X"]))
    neural = MLPClassifier(
        hidden_layer_sizes=tuple(config["neural_hidden_layers"]),
        activation="relu",
        solver="adam",
        alpha=config["neural_alpha"],
        batch_size=min(512, len(train_y)),
        learning_rate_init=config["neural_learning_rate"],
        shuffle=True,
        random_state=config["seed"],
        early_stopping=False,
    )
    best, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, config["neural_max_epochs"] + 1):
        neural.partial_fit(transformed, train_y, classes=[0, 1, 2])
        score = float(log_loss(valid_y, neural.predict_proba(valid_x), labels=[0, 1, 2]))
        trials.append({"model": "macro_neural", "epoch": epoch, "log_loss": score})
        if score < best_loss - 1e-5:
            best, best_loss, stale, best_epoch = copy.deepcopy(neural), score, 0, epoch
        else:
            stale += 1
        if stale >= config["neural_patience"]:
            break
    payload = neural_payload(best, imputer, scaler)
    if not np.allclose(
        predict_neural(payload, validation["X"]), best.predict_proba(valid_x), atol=1e-10
    ):
        raise PipelineError("Neural JSON artifact does not reproduce fitted probabilities")
    return {"logistic": logistic, "catboost": tree, "neural": payload}, {
        "catboost_rounds": rounds,
        "neural_epoch": best_epoch,
        "trials": trials,
    }


def predictions(components, x, threads=1):
    return {
        "macro_logistic": validate_probabilities(predict_linear(components["logistic"], x), len(x)),
        "macro_catboost": validate_probabilities(
            components["catboost"].predict_proba(x, thread_count=threads), len(x)
        ),
        "macro_neural": validate_probabilities(predict_neural(components["neural"], x), len(x)),
    }


def select_ensemble(predicted, labels, weights):
    arrays = list(predicted.values())
    selected, score = None, float("inf")
    for w in weights:
        if len(w) != len(arrays) or min(w) < 0 or not np.isclose(sum(w), 1):
            raise PipelineError("Ensemble weights must form a probability mixture")
        p = sum(weight * array for weight, array in zip(w, arrays, strict=True))
        loss = float(log_loss(labels, p, labels=[0, 1, 2]))
        if loss < score:
            selected, score = list(w), loss
    return selected, score


def mix(predicted, weights):
    return sum(w * p for w, p in zip(weights, predicted.values(), strict=True))
