import pandas as pd
import numpy as np
from numpy.random import normal
from numpy.random import uniform
from IMLearn.learners.regressors.polynomial_fitting import PolynomialFitting

X_domain = [-3.2,2.2]

if __name__ == '__main__':
    f_x = lambda x : (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    degrees = np.linspace(1,15,15)
    gaussian_noise = normal(loc=0,scale=1,size=1500)
    samples = uniform(low=X_domain[0],high=X_domain[1],size=1500)
    responses = np.add(np.apply_along_axis(f_x,0,samples), gaussian_noise)
    D = pd.DataFrame({"sample" : samples, "response" : responses})
    training_df = D.sample(n=1000)
    test_df = D.drop(training_df.index)

    # train the models
    models = []
    for degree in degrees:
        d_model = PolynomialFitting(int(degree))
        d_model._fit(training_df["sample"],training_df["response"])
        models.append(d_model)





