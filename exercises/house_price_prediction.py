from scipy import stats

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import loss_functions
import sys

sys.path.append("../")
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.default_format = "simple_white"

HOUSE_PRICES_PATH = "../datasets/house_prices.csv"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    panda_file = pd.read_csv(filename)

    # drop rows with nan
    panda_file = panda_file.dropna()

    # drop duplicated rows
    panda_file = panda_file.drop_duplicates()

    # drop row if the price is negative
    panda_file = panda_file.drop(panda_file[panda_file.price <= 0].index)

    panda_file = panda_file.drop(['id', 'date', 'long','lat'], axis=1)



    panda_file = pd.get_dummies(panda_file, prefix='zipcode_', columns=['zipcode'])




    y = pd.Series(panda_file['price'])
    X = panda_file.drop('price', axis=1)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = np.std(y, axis=0)

    for feature in X.keys():
        std = np.std(X[feature], ddof=1) * y_std
        cov = np.cov(y, X[feature], ddof=1)
        f_pearson_correlation = cov[0, 1] / std
        print(feature, f_pearson_correlation)
        fig = go.Figure([go.Scatter(y=y.values, x=X[feature].values,
                                    mode="markers", marker=dict(color="black", opacity=.7), showlegend=False)],
                        layout=go.Layout(title=f'scatter between {feature} and response - Pearson Correlation: {f_pearson_correlation} ',
                                         xaxis={"title": f"x - {feature}"},
                                         yaxis={"title": "y - Response"},
                                         height=400))
        # fig.show()
        fig.write_image(f"{output_path}/pearson.correlation.{feature}.png")

def test_func():

    pass


if __name__ == '__main__':
    np.random.seed(0)
    test_func()
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(HOUSE_PRICES_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)




    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    linear_reg = LinearRegression()
    loss_vec = []
    var_vec = []
    all_p = np.linspace(10, 100, 100).astype(int)
    for p in all_p:
        p_loss=[]
        for i in range(10):
            p_samples = pd.concat([train_X, train_y], ignore_index=True, axis=1).sample(frac=(p/100))
            linear_reg._fit(np.array(p_samples.iloc[:,:-1]),np.array(p_samples.iloc[:,-1]))
            p_loss.append(linear_reg._loss(np.array(test_X),(np.array(test_y))))
        loss_vec.append(np.mean(p_loss,axis=0))
        var_vec.append(np.std(p_loss,axis=0))
        print('loss:',loss_vec[-1],'std:',var_vec[-1])
    loss_vec = np.array(loss_vec)
    var_vec = np.array(var_vec)
    go.Figure([go.Scatter(x=all_p, y=loss_vec, mode="markers+lines",
                          name="Mean Prediction",
                          marker=dict(color="green", opacity=.7)),
                          go.Scatter(x=all_p, y=loss_vec-2*var_vec, fill=None,
                                     mode="lines", line=dict(color="lightgrey"),
                                     showlegend=False),
                          go.Scatter(x=all_p, y=loss_vec+2*var_vec, fill='tonexty',
                                     mode="lines", line=dict(color="lightgrey"),
                                     showlegend=False)],
                    layout=go.Layout(title="Average Loss As Function Of Training Size",
                                     xaxis_title='p - Training Size',
                                     yaxis_title='Average Loss')).show()


    # print("mean:",all_mean)
    # print("std:", all_var)



