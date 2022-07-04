from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from numpy.random import normal
from numpy.random import uniform
from IMLearn.learners.regressors.polynomial_fitting import PolynomialFitting


from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X_domain = [-1.2,2]
    np.random.seed(0)
    f_x = lambda x : (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    gaussian_noise = normal(loc=0,scale=noise,size=n_samples)
    samples = uniform(low=X_domain[0],high=X_domain[1],size=n_samples)
    orig_responses = np.apply_along_axis(f_x,0,samples)
    responses = np.add(orig_responses, gaussian_noise)
    D = pd.DataFrame({"sample" : samples, "response" : responses})
    train_data, train_response, test_data, test_response = split_train_test(pd.DataFrame(D["sample"]), D["response"],(2/3))
    plt.scatter(D["sample"],orig_responses, label='noiseless model')
    plt.scatter(train_data.squeeze(),train_response.squeeze(), label='training model')
    plt.scatter(test_data.squeeze(),test_response.squeeze(), label='testing model')
    plt.title("Scatter plot of predictions and their partitions")
    plt.legend()
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    folds = 5
    degrees = np.linspace(1,20,20)
    avg_errors = []
    for degree in degrees:
        estimator = PolynomialFitting(int(degree))
        degree_error = cross_validate(estimator,train_data.squeeze(),train_response.squeeze(),mean_square_error,folds)
        avg_errors.append(degree_error)
    assert len(avg_errors) == len(degrees)
    plt.bar(degrees -0.2, [avg_errors[i][0] for i in range(len(avg_errors))],width=0.4,label="training error")
    plt.bar(degrees +0.2, [avg_errors[i][1] for i in range(len(avg_errors))],width=0.4,label="validation error")
    plt.legend()
    plt.title("k-degree with training and validation error")
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    validation_errors = [errors[1] for errors in avg_errors]
    k_star = validation_errors.index(min(validation_errors))
    model = PolynomialFitting(k_star)
    model.fit(train_data.squeeze(), train_response.squeeze())
    print("k-star is -")
    print(k_star)
    print("Testing error :-")
    print(model.loss(test_data.squeeze(), test_response.squeeze()))




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X,y = datasets.load_diabetes(return_X_y=True)
    training_data = X[:n_samples]
    training_predictions = y[:n_samples]
    testing_data = X[n_samples:]
    testing_predictions = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lasso_lam_range = np.arange(0,20,20/500)
    ridge_lam_range = lasso_lam_range
    ridge_err_avg = []
    lasso_err_avg = []
    for i in range(500):
        ridge_estimator = RidgeRegression(ridge_lam_range[i])
        lasso_estimator = Lasso(lasso_lam_range[i])
        ridge_err_avg.append(cross_validate(ridge_estimator,training_data, training_predictions,mean_square_error,5))
        lasso_err_avg.append(cross_validate(lasso_estimator,training_data, training_predictions,mean_square_error,5))

    ridge_train_loss = [losses[0] for losses in ridge_err_avg]
    ridge_validation_loss = [losses[1] for losses in ridge_err_avg]
    lasso_train_loss = [losses[0] for losses in lasso_err_avg]
    lasso_validation_loss = [losses[1] for losses in lasso_err_avg]

    fig = go.Figure([go.Scatter(x=ridge_lam_range, y=ridge_train_loss, name="ridge train loss"),
                         go.Scatter(x=ridge_lam_range, y=ridge_validation_loss, name="ridge validation loss"),
                         go.Scatter(x=lasso_lam_range, y=lasso_train_loss, name="lasso train loss"),
                         go.Scatter(x=lasso_lam_range, y=lasso_validation_loss, name="lasso validation loss")])
    fig.update_layout(title_text="Ridge and Lasso errors as function of lambda")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_star =  ridge_lam_range[ridge_validation_loss.index(min(ridge_validation_loss))]
    lasso_star = lasso_lam_range[lasso_validation_loss.index(min(lasso_validation_loss))]
    print(f"Lasso star : {lasso_star}")
    print(f"Ridge star : {ridge_star}")

    ridge_estimator = RidgeRegression(ridge_star)
    lasso_estimator = Lasso(lasso_star)
    least_squares = LinearRegression()

    ridge_estimator.fit(training_data, training_predictions)
    lasso_estimator.fit(training_data, training_predictions)
    least_squares.fit(training_data, training_predictions)
    print(f"Loss over the lasso model : {mean_square_error(lasso_estimator.predict(testing_data),testing_predictions)}")
    print(f"Loss over the Ridge model : {ridge_estimator.loss(testing_data, testing_predictions)}")
    print(f"Loss over Least Squares : {least_squares.loss(testing_data, testing_predictions)}")

if __name__ == '__main__':
    # select_polynomial_degree(1500,10)
    select_regularization_parameter()
