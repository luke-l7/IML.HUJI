import matplotlib.pyplot as plt
import numpy
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_frame: pd.DataFrame = pd.read_csv(filename, parse_dates=['Date'])
    data_frame = data_frame.dropna()
    data_frame = data_frame.drop_duplicates()
    # drop neg values
    data_frame = data_frame[data_frame.select_dtypes(include=[np.number]).ge(0).all(1)]
    # create DOY
    data_frame['DayOfYear'] = pd.to_datetime (data_frame['Date']).dt.dayofyear
    return data_frame


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("..\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country

    # select rows that are specified for Israel
    israel_data : pd.DataFrame = df.loc[df['Country'] == "Israel"]
        # define index column
    israel_data.set_index('DayOfYear', inplace = True)

        # group data by product and display sales as line chart
    israel_data.groupby('Year')['Temp'].plot.line(legend=True,title="Temperature by the Day of year, plotted by year",style='o', markeredgecolor='white')

    plt.show()
        # plot by month
    israel_data.groupby('Month')['Temp'].agg('std').plot.bar(legend = True, rot=0, title="SD of temprature in each month")

    plt.show()

    # Question 3 - Exploring differences between countries

    grouped = df.groupby(["City","Month"])
    grouped = grouped.agg({'Temp' : ["mean","std"]}).reset_index()
    grouped.columns = ["City","Month","mean","std"]
    px.line(grouped,x="Month",y="mean",color=grouped["City"].astype(str),title="mean for each country",error_y="std").show()

    # Question 4 - Fitting model for different values of `k`
    israel_y = israel_data['Temp']
    israel_data.drop(columns=['Temp'],inplace=True)
    degrees = numpy.arange(1,11,dtype=int)
    losses = []
    train_d,train_y,test_d,test_y = split_train_test(israel_data,israel_y)
    for k in degrees:
        poly : PolynomialFitting = PolynomialFitting(k)
        poly.fit(np.array(train_d['DayOfYear']),np.array(train_y))
        losses.append(round(poly.loss(np.array(test_d['DayOfYear']),np.array(test_y)),2))
    print(losses)
    px.bar(pd.DataFrame.join(pd.DataFrame(degrees,columns=['degrees']),pd.DataFrame(losses,columns=['losses'])),x='degrees',y='losses',color='degrees').show()

    # Question 5 - Evaluating fitted model on different countries

    poly : PolynomialFitting = PolynomialFitting(5)
    poly.fit(np.array(israel_data['DayOfYear']),np.array(israel_data['Temp']))
    losses = []
    countries = df["Country"].unique()
    for country in countries:
        country_data = df.loc[df['Country'] == country]
        country_y = country_data['Temp']
        losses.append(poly.loss(country_data['DayOfYear'],country_y))
    print(losses)
    px.bar(x=countries,y=losses,color=countries,title="Error value predicted for other countries with Israel's model").show()



