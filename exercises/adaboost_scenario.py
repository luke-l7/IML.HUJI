import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)
    iterations = np.arange(1,n_learners)
    train_losses = []
    test_losses = []
    for i in iterations:
        train_loss = ada.partial_loss(train_X, train_y, i)
        test_loss = ada.partial_loss(test_X, test_y, i)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    fig = go.Figure([
        go.Scatter(x=iterations, y=train_losses, mode='markers+lines', name=r'Train loss'),
               go.Scatter(x=iterations, y=test_losses, mode='markers+lines', name=r'Test loss')],
              layout=go.Layout(
                  title=r"$\text{training and test errors as a function of iterations}$",
                  xaxis_title="iterations", yaxis_title="loss", height=800))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"{t} iterations" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    def helper_predict(x):
        return ada.partial_predict(x, t)

    for i, t in enumerate(T):
        fig.add_traces([decision_surface(helper_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol="x", colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=0.5)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"Decision boundary obtained by using the the ensemble up to t iteration",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble

    # Question 4: Decision surface with weighted samples


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # fit_and_evaluate_adaboost(0.4)