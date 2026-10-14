from pathlib import Path
from tempfile import TemporaryDirectory

import xgboost as xgb
from catboost import CatBoostClassifier
from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.storage.archive import digest
from stock_predictor.training.metrics import validate_probabilities


def fit_model(family, train_x, train_y, valid_x, valid_y, candidate, config, rounds=None):
    if family == "xgboost":
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cpu",
            "max_bin": 128,
            "learning_rate": config["learning_rate"],
            "nthread": config["threads"],
            "seed": config["seed"],
            "colsample_bytree": 0.9,
            **{k: v for k, v in candidate.items() if k != "name"},
        }
        train = xgb.DMatrix(train_x, label=train_y, nthread=config["threads"])
        valid = (
            xgb.DMatrix(valid_x, label=valid_y, nthread=config["threads"])
            if valid_x is not None
            else None
        )
        model = xgb.train(
            params,
            train,
            num_boost_round=rounds or config["max_rounds"],
            evals=[(valid, "validation")] if valid is not None else [],
            early_stopping_rounds=config["early_stopping_rounds"] if valid is not None else None,
            verbose_eval=False,
        )
        selected_rounds = (
            model.best_iteration + 1 if valid is not None else model.num_boosted_rounds()
        )
        return model[:selected_rounds], int(selected_rounds)
    if family != "catboost":
        raise PipelineError("Unsupported challenger model family")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        classes_count=3,
        iterations=rounds or config["max_rounds"],
        learning_rate=config["learning_rate"],
        random_seed=config["seed"],
        thread_count=config["threads"],
        allow_writing_files=False,
        nan_mode="Min",
        verbose=False,
        **{k: v for k, v in candidate.items() if k != "name"},
    )
    kwargs = (
        {
            "eval_set": (valid_x, valid_y),
            "early_stopping_rounds": config["early_stopping_rounds"],
            "use_best_model": True,
        }
        if valid_x is not None
        else {"use_best_model": False}
    )
    model.fit(train_x, train_y, **kwargs)
    if list(model.classes_) != [0, 1, 2]:
        raise PipelineError("CatBoost direction class order differs from DOWN/FLAT/UP")
    return model, int(model.tree_count_)


def predict_model(family, model, x, threads=1):
    if list(x) != FEATURE_COLUMNS:
        raise PipelineError("Challenger feature ordering differs from shared schema")
    if family == "xgboost":
        probabilities = model.predict(xgb.DMatrix(x, nthread=threads))
    else:
        probabilities = model.predict_proba(x, thread_count=threads)
    return validate_probabilities(probabilities, len(x))


def save_model(archive, family, name, model):
    if family == "xgboost":
        body, suffix = bytes(model.save_raw(raw_format="ubj")), "ubj"
    else:
        with TemporaryDirectory(prefix="stock-challenger-") as directory:
            path = Path(directory) / "model.cbm"
            model.save_model(str(path), format="cbm")
            body = path.read_bytes()
        suffix = "cbm"
    return archive.write(f"models/challengers/{family}/{name}/{digest(body)}.{suffix}", body)


def load_model(archive, family, ref):
    body = archive.read(ref["key"], sha256=ref["sha256"])
    if body is None:
        raise PipelineError("Challenger artifact is missing")
    if family == "xgboost":
        model = xgb.Booster(model_file=bytearray(body), params={"nthread": 1})
        names = model.feature_names
    else:
        model = CatBoostClassifier()
        model.load_model(blob=body, format="cbm")
        names = model.feature_names_
    if names != FEATURE_COLUMNS:
        raise PipelineError("Native challenger artifact has the wrong feature schema")
    return model
