import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stock_predictor.config import PipelineError
from stock_predictor.replay import read_json
from stock_predictor.training.bundle import save_json_model
from stock_predictor.training.estimators import linear_payload, predict_linear
from stock_predictor.training.metrics import encode_labels, validate_probabilities
from stock_predictor_challengers.models import fit_model, save_model


def fit(train, valid, config):
    train_y, valid_y = (encode_labels(b["y"].y_direction_5d) for b in (train, valid))
    best, best_loss, trials = None, float("inf"), []
    for c in config["logistic_cs"]:
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=c, max_iter=1500, solver="lbfgs", random_state=config["seed"]),
        )
        model.fit(train["X"], train_y)
        if list(model[-1].classes_) != [0, 1, 2]:
            raise PipelineError("Direction fitting must include every class")
        score = float(log_loss(valid_y, model.predict_proba(valid["X"]), labels=[0, 1, 2]))
        trials.append({"family": "logistic", "C": c, "log_loss": score})
        if score < best_loss:
            best_loss = score
            best = linear_payload(model, "logistic", list(train["X"]))
            if not np.allclose(
                predict_linear(best, valid["X"]), model.predict_proba(valid["X"]), atol=1e-10
            ):
                raise PipelineError("Fundamental/news logistic JSON changed probabilities")
    tree, rounds = fit_model(
        "catboost", train["X"], train_y, valid["X"], valid_y, config["catboost"], config
    )
    models = {"logistic": best, "catboost": tree}
    p = predict(models, valid["X"], config["threads"])
    losses = {
        name: float(log_loss(valid_y, values, labels=[0, 1, 2])) for name, values in p.items()
    }
    selected = min(losses, key=losses.get)
    return models, {
        "family": selected,
        "tuning_log_loss": losses,
        "catboost_rounds": rounds,
        "trials": trials,
    }


def predict(models, x, threads=1):
    if (
        list(x) != models["logistic"]["feature_columns"]
        or list(x) != models["catboost"].feature_names_
    ):
        raise PipelineError("Fundamental/news model feature ordering differs")
    return {
        "logistic": validate_probabilities(predict_linear(models["logistic"], x), len(x)),
        "catboost": validate_probabilities(
            models["catboost"].predict_proba(x, thread_count=threads), len(x)
        ),
    }


def save(archive, models, name):
    refs = {
        "logistic": save_json_model(archive, "fundamental-news/" + name, models["logistic"]),
        "catboost": save_model(archive, "catboost", "fundamental-news-" + name, models["catboost"]),
    }
    reloaded = {"logistic": read_json(archive, refs["logistic"]), "catboost": CatBoostClassifier()}
    ref = refs["catboost"]
    body = archive.read(ref["key"], sha256=ref["sha256"])
    if body is None:
        raise PipelineError("Missing fundamental/news tree artifact")
    reloaded["catboost"].load_model(blob=body, format="cbm")
    return refs, reloaded
