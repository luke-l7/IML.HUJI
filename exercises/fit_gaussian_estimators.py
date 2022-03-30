import itertools

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu = 10
    sigma = 1
    # Question 1 - Draw samples and print fitted model
    m=1000 # number of samples
    samples = np.random.normal(mu,sigma,size=m)
    gaussian_param = UnivariateGaussian()
    gaussian_param.fit(samples)
    print(gaussian_param.mu_,gaussian_param.var_)

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    ms = np.arange(10,1010,10)
    for m in ms:
        X = np.random.normal(mu, sigma, size=m)
        estimated_mean.append(np.mean(X))
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$'),
               go.Scatter(x=ms, y=[mu]*len(ms), mode='lines', name=r'$\mu$')],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    PDFs = gaussian_param.pdf(samples)
    go.Figure([go.Histogram(x=samples, opacity=0.75, bingroup=1, histnorm='probability density', marker_color="rgb(219,124,134)", name=r'$\hat\mu_1$')],
              layout=go.Layout(barmode='overlay',
                               title=r"$\text{(8) Mean estimator distribution}$",
                               xaxis_title="r$\hat\mu$",
                               yaxis_title="density",
                               height=300)).show()
    print(UnivariateGaussian.log_likelihood(mu,sigma,samples))


def test_multivariate_gaussian():
    m = 1000
    mean = np.array([0,0,4,0])
    sigma_matrix = np.array([1,0.2,0,0.5,0.2,2,0,0,0,0,1,0,0.5,0,0,1]).reshape(4,4)
    multi_gaussian_var = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(mean,sigma_matrix,m)
    multi_gaussian_var.fit(samples)
    print(multi_gaussian_var.mu_)
    print( multi_gaussian_var.cov_)

    # Question 5 - Likelihood evaluation
    lin_space = np.linspace(-10,10,200)
    log_likelihoods=[]
    for f1 in lin_space:
        for f3 in lin_space:
            mu = np.array([f1,0,f3,0])
            samples = np.random.multivariate_normal(mu,sigma_matrix,m)
            log_likelihoods.append(MultivariateGaussian.log_likelihood(mu,sigma_matrix,samples))

    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
