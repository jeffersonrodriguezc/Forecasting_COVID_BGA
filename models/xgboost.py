import os
import numpy as np
import pandas as pd
import xgboost as xgb

from glob import glob
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from Forecasting_COVID_BGA.utils import sort_nicely

# MAPE computation
def mape(y, yhat, perc=False):
    n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)
    mape = []
    for a, f in zip(y, yhat):
        # avoid division by 0
        if f > 1e-9:
            mape.append(np.abs((a - f) / a))
    mape = np.mean(np.array(mape))
    return mape * 100. if perc else mape


# mape_scorer = make_scorer(mape, greater_is_better=False)
mape_scorer = make_scorer(mean_squared_error)


def train_xgb(params, X_train, y_train):
    """
    Train XGBoost regressor using the parameters given as input. The model
    is validated using standard cross validation technique adapted for time series
    data. This function returns a friendly output for the hyperopt parameter optimization
    module.
    
    Parameters
    ----------
    params: dict with the parameters of the XGBoost regressor. For complete list see: 
            https://xgboost.readthedocs.io/en/latest/parameter.html
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets    
    
    Returns
    -------
    dict with keys 'model' for the trained model, 'status' containing the hyperopt
    status string and 'loss' with the RMSE obtained from cross-validation
    """

    n_estimators = int(params["n_estimators"])
    max_depth = int(params["max_depth"])

    try:
        model = xgb.XGBRegressor(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 learning_rate=params["learning_rate"],
                                 subsample=params["subsample"],
                                 objective='reg:squarederror')

        result = model.fit(X_train,
                           y_train,
                           eval_set=[(X_train, y_train)],
                           early_stopping_rounds=50,
                           verbose=False)

        # cross validate using the right iterator for time series
        cv_space = TimeSeriesSplit(n_splits=5)
        cv_score = cross_val_score(model,
                                   X_train,
                                   y_train,
                                   cv=cv_space,
                                   scoring=mape_scorer)

        rmse = np.abs(np.mean(np.array(cv_score)))
        return {"loss": rmse, "status": STATUS_OK, "model": model}

    except ValueError as ex:
        return {"error": ex, "status": STATUS_FAIL}


def optimize_xgb(X_train, y_train, max_evals=10):
    """
    Run Bayesan optimization to find the optimal XGBoost algorithm
    hyperparameters.
    
    Parameters
    ----------
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets
    max_evals: the maximum number of iterations in the Bayesian optimization method
    
    Returns
    -------
    best: dict with the best parameters obtained
    trials: a list of hyperopt Trials objects with the history of the optimization
    """

    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
        "max_depth": hp.quniform("max_depth", 1, 8, 1),
        "learning_rate": hp.loguniform("learning_rate", -5, 1),
        "subsample": hp.uniform("subsample", 0.8, 1),
        "gamma": hp.quniform("gamma", 0, 100, 1)
    }

    objective_fn = partial(train_xgb, X_train=X_train, y_train=y_train)

    trials = Trials()
    best = fmin(fn=objective_fn,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    # evaluate the best model on the test set
    print(f"""
    Best parameters:
        learning_rate: {best["learning_rate"]} 
        n_estimators: {best["n_estimators"]}
        max_depth: {best["max_depth"]}
        sub_sample: {best["subsample"]}
        gamma: {best["gamma"]}
    """)
    return best, trials


def train_direct_xgb(x_train_dset, y_train_dset, params):
    print('Training XGBoosts models using a direct approach...')
    models_res = []
    for idx in range(len(x_train_dset)):
        print('Training XGBoost model {}...'.format(idx))
        res = train_xgb(params[idx], x_train_dset[idx], y_train_dset[idx])
        models_res.append(res)

    return models_res


def optimize_direct_xgb(x_dset, y_dset, max_evals=10):
    params, trials = [], []
    print('Optimizing XGBoosts models using a direct approach...')
    for idx, (X_train, y_train) in enumerate(zip(x_dset, y_dset)):
        print('Finding best params for XGBoost model {}...'.format(idx))
        best_params, best_trials = optimize_xgb(X_train, y_train, max_evals=max_evals)
        params.append(best_params)
        trials.append(best_trials)

    return params, trials

def predict_direct_xgb(X_test, models):
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test))
    return np.hstack(predictions)


def load_direct_model(source_dir):
    model_paths = glob(os.path.join(source_dir, '*'))
    model_paths.pop(model_paths.index(source_dir+'scaler.pkl'))
    sort_nicely(model_paths)
    models = []
    for model_path in model_paths:
        trained_xgb = xgb.XGBRegressor()
        trained_xgb.load_model(model_path)
        models.append(trained_xgb)
    return models
