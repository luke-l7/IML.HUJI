import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.mean((y_true-y_pred)**2)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    if normalize:
        y_true = (1/np.linalg.norm(y_true)) * y_true
        y_pred = (1/np.linalg.norm(y_pred)) * y_pred
    s = [y_t * y_p <= 0 for (y_t,y_p) in zip(y_true,y_pred)]
    return np.count_nonzero(s)



def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    positives = sum(x > 0 for x in y_pred)
    negatives = len(y_pred) - positives
    t_positives = sum(x > 0 and x == y for (x,y) in zip(y_true,y_pred))
    t_negatives = sum(x <= 0 and x == y for (x,y) in zip(y_true,y_pred))
    return (t_positives + t_negatives)/(positives + negatives)



def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
