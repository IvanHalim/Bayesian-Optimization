Portfolio Allocation using Bayesian Optimization
================

The purpose of this project is to gain a deeper understanding of
Bayesian Optimization and its practical application in data analysis and
simulation. Bayesian Optimization is an increasingly popular topic in
the field of Machine Learning. It allows us to find an optimal
hyperparameter configuration for a particular machine learning algorithm
without too much human intervention. Bayesian Optimization has several
advantages compared to other optimization algorithm. The first advantage
of Bayesian Optimization is that it does not require hand-tuning or
expert knowledge, which makes it easily scalable for larger, more
complicated analysis. The second advantage of Bayesian Optimization is
when evaluations of the fitness function are expensive to perform. If
the fitness function \(f\) is cheap to evaluate we could sample at many
points e.g. via grid search, random search or numeric gradient
estimation. However, if function evaluation is expensive e.g. tuning
hyperparameters of a deep neural network, probe drilling for oil at
given geographic coordinates or evaluating the effectiveness of a drug
candidate taken from a chemical search space then it is important to
minimize the number of samples drawn from the black box function \(f\).

## Project Organization

  - The `analysis` folder contains Rmarkdown files (along with knitted
    versions for easy viewing) with the code used to run simulations and
    analyze and visualize the results.

  - The `data` folder contains the New York Stock Exchange dataset used
    for this simulation. Data is imported from Kaggle (
    <https://www.kaggle.com/dgawlik/nyse> ).

  - The `R` folder contains the R functions used to run Bayesian
    Optimization

  - The `reports` folder contains deliverables such as project proposal
    and final report.

  - The `results` folder contains files generated files generated during
    clean-up and analysis as well as the final result of the simulation.

  - The `man` folder contains documentation for the functions defined in
    the `R` folder. Documentation for each function can be rendered
    using the standard R syntax (e.g. `?function`).

## Gaussian Process

Gaussian Process is a probabilistic model to approximate based on a
given set of data points. Gaussian Process models a function as a set of
random variables whose joint distribution is a multivariate normal
distribution, with a specific mean vector and covariance
matrix.

![equation](https://latex.codecogs.com/gif.latex?f%28x_%7B1%3Ak%7D%29%20%5Csim%20%5Cmathcal%7BN%7D%28%5Cmu%28x_%7B1%3Ak%7D%29%2C%20%5CSigma%28x_%7B1%3Ak%7D%2Cx_%7B1%3Ak%7D%29%29)

Where,

  - ![equation](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D%28x%2Cy%29):
    Gaussian/Normal random
    distribution
  - ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%28x_%7Bi%3Ak%7D%29):
    Mean vector of each
    ![equation](https://latex.codecogs.com/gif.latex?f%28x_i%29)
  - ![equation](https://latex.codecogs.com/gif.latex?%5CSigma%28x_%7Bi%3Ak%7D%2C%20x_%7Bi%3AK%7D%29):
    Covariance matrix of each pair of
    ![equation](https://latex.codecogs.com/gif.latex?f%28x_i%29)

For a candidate point \(x'\), its function value \(f(x')\) can be
approximated, given a set of observed values \(f(x_{1:n})\), using the
posterior
distribution,

![equation](https://latex.codecogs.com/gif.latex?f%28x%29%7Cf%28x_%7B1%3An%7D%29%20%5Csim%20%5Cmathcal%7BN%7D%28%5Cmu_n%28x%29%2C%20%5Csigma_n%5E2%28x%29%29)

Where,

  - ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_n%28x%29%20%3D%20%5CSigma_0%28x%2Cx_%7Bi%3An%7D%29%20%5Cast%20%5CSigma_0%28x_%7Bi%3An%7D%2Cx_%7Bi%3An%7D%29%5E%7B-1%7D%20%5Cast%20%28f%28x_%7B1%3An%7D%29%20-%20%5Cmu_0%28x_%7B1%3An%7D%29%29%20+%20%5Cmu_0%28x%29)
  - ![equation](https://latex.codecogs.com/gif.latex?%5Csigma_n%5E2%28x%29%20%3D%20%5CSigma_0%28x%2Cx%29%20-%20%5CSigma_0%28x%2C%20x_%7Bi%3An%7D%29%20%5Cast%20%5CSigma_0%28x_%7Bi%3An%7D%2Cx_%7Bi%3An%7D%29%5E%7B-1%7D%20%5Cast%20%5CSigma_0%28x_%7Bi%3An%7D%2Cx%29)

Below is the example of Gaussian Process posterior over function graph.
The following example draws three samples from the posterior and plots
them along with the mean, confidence interval and training data.

``` r
noise <- 0.4
gpr <- gpr.init(sigma_y = noise)

# Finite number of points
X <- seq(-5, 5, 0.2)

# Noisy training data
X_train <- seq(-3, 3, 1)
Y_train <- sin(X_train) + noise * rnorm(n = length(X_train))
gpr <- gpr.fit(X_train, Y_train, gpr)

# Compute mean and covariance of the posterior predictive distribution
result <- gpr.predict(X, gpr)
mu_s <- result$mu_s
cov_s <- result$cov_s

samples <- mvrnorm(n = 3, mu = mu_s, Sigma = cov_s)
plot_gp(mu_s, cov_s, X, X_train, Y_train, samples)
```

![](README_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->
