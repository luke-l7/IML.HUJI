import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


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
    unwanted_features = ["id",
                         "date",
                         "yr_renovated",
                         "lat",
                         "long"
                         ]
    data_frame: pd.DataFrame = pd.read_csv(filename)

    # remove unwanted features
    for feature in unwanted_features:
        data_frame.drop(feature, axis=1, inplace=True)
    # adjust faulty data
    dropped_rows = []
    for index, row in data_frame.iterrows():
        if True in [x <= 0 for x in row]:
            dropped_rows.append(index)
    data_frame.drop(dropped_rows, axis=0)
    data_frame = data_frame.dropna()
    data_frame = data_frame.drop_duplicates()
    # other pre-processing
    data_frame['yr_built'] = data_frame['yr_built'].map(lambda x: 2022 - x)
    # hot encode zipcodes
    encoded_data = pd.get_dummies(data_frame.zipcode, prefix='zipcode')
    data_frame = pd.concat([data_frame, encoded_data], axis=1)
    # remove zipcode column, and price column
    data_frame.drop(['zipcode'], axis=1, inplace=True)
    Y: pd.Series = data_frame['price']
    data_frame.drop(['price'], axis=1, inplace=True)
    return data_frame, Y


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
    import matplotlib.pyplot as plt
    # import seaborn as sns

    # remove categorical data
    orig_df = X.copy(deep=True)
    orig_df.drop(columns=orig_df.columns[14:], inplace=True)

    # join with solution
    orig_df = pd.DataFrame.join(pd.DataFrame(y), orig_df)
    cov_array = orig_df.cov()
    features_evaluation_cov : np.array = cov_array['price'][:15] # take only numerical data
    response_var = Y.std()
    i = 0
    for (feature_name,feature_data) in orig_df.iteritems():
        if i == 15:
            break
        features_evaluation_cov[i] /= (feature_data.std() * response_var)
        i += 1
    for (feature_name,feature_data) in orig_df.iteritems():
        orig_df.hist(column=feature_name, bins=50)
        plt.savefig(output_path + "\\" + feature_name)

    print(features_evaluation_cov)


if __name__ == '__main__' :
    from IMLearn.metrics.loss_functions import mean_square_error
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print(mean_square_error(y_true, y_pred))


    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, Y = load_data("..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    print(feature_evaluation(df, Y))

    # Question 3 - Split samples into training- and testing sets.
    train_d,train_y,test_d,test_y = split_train_test(df,Y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_model = LinearRegression()
    var_mean_for_percentage = []
    percents = np.linspace(10,100,91)
    mean_arr = []
    var_arr = []

    for percentage in percents:
        loss_arr = []
        for i in range(10):
            train_d,train_y,test_d,test_y = split_train_test(df,Y,percentage/100)
            linear_model.fit(np.array(train_d),np.array(train_y))
            loss = linear_model.loss(np.array(test_d).reshape(test_d.shape),np.array(test_y).reshape(test_y.shape))
            loss_arr.append(loss)
        series = pd.Series(loss_arr)
        mean_arr.append(series.mean())
        var_arr.append(series.std())
    mean_arr = mean_arr[::-1]
    var_arr = var_arr[::-1]
    mean_arr = np.array(mean_arr)
    var_arr = np.array(var_arr)

    fig = go.Figure(
        (go.Scatter(x=percents, y=mean_arr, mode="markers+lines", name="mean loss prediction", line=dict(dash="dash"),
                    marker=dict(color="green", opacity=.7), ),
         go.Scatter(x=percents, y=mean_arr - (2 * var_arr), fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=percents, y=mean_arr + (2 * var_arr), fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),))
    st = "MSE and STDDS of losses over house prices, as a function of train data percentage"
    fig.update_layout(title=dict({'text': st}))
    fig.show()







