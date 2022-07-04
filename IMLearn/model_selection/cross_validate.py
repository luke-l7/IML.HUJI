from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # train_scores = []
    # validation_scores = []
    # data_folds = np.array_split(X, cv)
    # predictions_folds = np.array_split(y,cv)
    # for fold_index in range(len(data_folds)):
    #     test_fold = fold_index
    #     training_X = X.copy()
    #     training_y = y.copy()
    #     c = 0
    #     for sample in training_X:
    #         if sample in data_folds[test_fold]:
    #             c += 1
    #     print(c)
    #     print(len(data_folds[test_fold]))
    #     assert c == len(data_folds[test_fold])
    #
    #     c = 0
    #     for prediction in training_y:
    #         if prediction in predictions_folds[test_fold]:
    #             c += 1
    #     print(c)
    #     assert c == len(predictions_folds[test_fold])
    #     training_X = training_X[~training_X.isin(data_folds[test_fold])]
    #     training_y = training_y[~training_y.isin(predictions_folds[test_fold])]
    #     train_scores.append(scoring(estimator.predict(training_X),training_y))
    #     validation_scores.append(scoring(estimator.predict(data_folds[test_fold]),predictions_folds[test_fold]))
    # return np.average(train_scores), np.average(validation_scores)
    training_err = []
    validation_err = []
    folds = np.remainder(np.arange(X.shape[0]), cv)

    for k in range(cv):
        train_data = X[folds != k]
        train_predictions = y[folds != k]
        validation_data = X[folds == k]
        validation_predictions = y[folds == k]
        estimator.fit(train_data, train_predictions)
        training_err.append(scoring(estimator.predict(train_data), train_predictions))
        validation_err.append(scoring(estimator.predict(validation_data), validation_predictions))
    return np.average(training_err), np.average(validation_err)
