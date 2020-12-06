Fitness Function
================
Ivan Timothy Halim
12/3/2020

# Portfolio Optimization

Portfolio optimization problem is concerned with managing the portfolio
of assets that minimizes the risk objectives subjected to the constraint
for guaranteeing a given level of returns. One of the fundamental
principles of financial investment is diversification where investors
diversify their investments into different types of assets. Portfolio
diversification minimizes investors exposure to risks, and maximizes
returns on portfolios.

The fitness function is the adjusted Sharpe Ratio for restricted
portofolio, which combines the information from mean and variance of an
asset and functioned as a risk-adjusted measure of mean return, which is
often used to evaluate the performance of a portfolio.

The Sharpe ratio can help to explain whether a portfolio’s excess
returns are due to smart investment decisions or a result of too much
risk. Although one portfolio or fund can enjoy higher returns than its
peers, it is only a good investment if those higher returns do not come
with an excess of additional risk.

The greater a portfolio’s Sharpe ratio, the better its risk-adjusted
performance. If the analysis results in a negative Sharpe ratio, it
either means the risk-free rate is greater than the portfolio’s return,
or the portfolio’s return is expected to be negative.

The fitness function is shown below:

![equation](https://latex.codecogs.com/gif.latex?%5Cmax%20f%28x%29%20%3D%20%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5EN%20W_i%20%5Cast%20r_i%20-%20R_f%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5EN%20%5Csum%5Climits_%7Bj%3D1%7D%5EN%20W_i%20%5Cast%20W_j%20%5Cast%20%5Csigma_%7Bij%7D%7D)

**Subject To**

![equation](https://latex.codecogs.com/gif.latex?%5Csum%5Climits_%7Bi%3D1%7D%5EN%20W_i%20%3D%201)

![equation](https://latex.codecogs.com/gif.latex?0%20%5Cleq%20W_i%20%5Cleq%201)

![equation](https://latex.codecogs.com/gif.latex?i%20%3D%201%2C2%2C...%2CN)

-   ![equation](https://latex.codecogs.com/gif.latex?N): Number of
    different assets
-   ![equation](https://latex.codecogs.com/gif.latex?W_i): Weight of
    each stock in the portfolio
-   ![equation](https://latex.codecogs.com/gif.latex?r_i): Return of
    stock ![equation](https://latex.codecogs.com/gif.latex?i)
-   ![equation](https://latex.codecogs.com/gif.latex?R_f): The test
    available rate of return of a risk-free security (i.e. the interest
    rate on a three month U.S. Treasury bill)
-   ![equation](https://latex.codecogs.com/gif.latex?%5Csigma_%7Bij%7D):
    Covariance between returns of assets
    ![equation](https://latex.codecogs.com/gif.latex?i) and
    ![equation](https://latex.codecogs.com/gif.latex?j)

Adjusting the portfolio weights
![equation](https://latex.codecogs.com/gif.latex?w_i), we can maximize
the portfolio Sharpe Ratio in effect balancing the trade-off between
maximizing the expected return and at the same time minimizing the risk.

## Define Fitness Function

Let’s define the fitness function. We will penalize the solution that
violate the constraint. Higher penalty will increase accuracy and force
the fitness value to get closer to the feasible area.

``` r
sharpe_ratio <- function(W, noise=0) {
  
  if (is.null(dim(W))) {
    W <- t(W)
  }
  
  # Calculate the numerator
  f1 <- W %*% mean_stock$mean
  
  # Calculate the denominator
  f2 <- numeric(length = nrow(W))
  for (i in 1:nrow(W)) {
    f2[i] <- sum(outer(W[i,], W[i,]) * nyse_cov)
  }
  
  # Calculate Fitness Value
  fitness <- (f1 - rf)/f2
  
  # Penalize Constraint Violation
  fitness <- fitness - 1e9 * (round(rowSums(W), 10) - 1)^2
  
  # Add noise
  fitness <- fitness + noise * rnorm(length(fitness))
  
  as.vector(fitness)
}
```

Let’s save our fitness function for future use,

``` r
saveRDS(sharpe_ratio, here("results", "sharpe_ratio_fn.rds"))
```
