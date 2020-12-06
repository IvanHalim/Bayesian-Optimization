Gaussian Process
================
Ivan Timothy Halim
12/2/2020

# Gaussian processes

## Introduction

In supervised learning, we often use parametric models
![equation](https://latex.codecogs.com/gif.latex?p%28y%7CX%2C%5Ctheta%29)
to explain data and infer optimal values of parameter
![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) via maximum
likelihood or maximum a posteriori estimation. If needed we can also
infer a full posterior distribution
![equation](https://latex.codecogs.com/gif.latex?p%28%5Ctheta%7CX%2Cy%29)
instead of a point estimate
![equation](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Ctheta%7D).
With increasing data complexity, models with a higher number of
parameters are usually needed to explain data reasonably well. Methods
that use models with a fixed number of parameters are called parametric
methods.

In non-parametric methods, on the other hand, the number of parameters
depend on the dataset size. For example, in Nadaraya-Watson kernel
regression, a weight
![equation](https://latex.codecogs.com/gif.latex?w_i) is assigned to
each observed target
![equation](https://latex.codecogs.com/gif.latex?y_i) and for predicting
the target value at a new point
![equation](https://latex.codecogs.com/gif.latex?x) a weighted average
is computed:

![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20w_i%20%28x%29%20y_i)

![equation](https://latex.codecogs.com/gif.latex?w_i%28x%29%20%3D%20%5Cfrac%7B%5Ckappa%28x%2Cx_i%29%7D%7B%5Csum_%7Bi%27%3D1%7D%5EN%20%5Ckappa%28x%2Cx_%7Bi%27%7D%29%7D)

Observations that are closer to
![equation](https://latex.codecogs.com/gif.latex?x) have a higher weight
than observations that are further away. Weights are computed from
![equation](https://latex.codecogs.com/gif.latex?x) and observed
![equation](https://latex.codecogs.com/gif.latex?x_i) with a kernel
![equation](https://latex.codecogs.com/gif.latex?%5Ckappa). A special
case is k-nearest neighbor (KNN) where the
![equation](https://latex.codecogs.com/gif.latex?k) closest observations
have a weight ![equation](https://latex.codecogs.com/gif.latex?1/k), and
all others have weight
![equation](https://latex.codecogs.com/gif.latex?0). Non-parametric
methods often need to process all training data for prediction and are
therefore slower at inference time than parametric methods. On the other
hand, training is usually faster as non-parametric models only need to
remember training data.

Another example of non-parametric methods are Gaussian processes (GPs).
Instead of inferring a distribution over the parameters of a parametric
function Gaussian processes can be used to infer a distribution over
functions directly. A Gaussian process defines a prior over functions.
After having observed some function values it can be converted into a
posterior over functions. Inference of continuous function values in
this context is known as GP regression but GPs can also be used for
classification.

A Gaussian process is a random process where any point
![equation](https://latex.codecogs.com/gif.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ed)
is assigned a random variable
![equation](https://latex.codecogs.com/gif.latex?f%28x%29) and where the
joint distribution of a finite number of these variables
![equation](https://latex.codecogs.com/gif.latex?p%28f%28x_1%29%2C...%2Cf%28x_N%29%29)
is itself Gaussian:

![equation](https://latex.codecogs.com/gif.latex?p%28f%7CX%29%20%3D%20%5Cmathcal%7BN%7D%28f%7C%5Cmu%2C%20K%29)

Where,

-   ![equation](https://latex.codecogs.com/gif.latex?f%20%3D%20%28f%28x_1%29%2C...%2Cf%28x_N%29%29 "fig:")
-   ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20%3D%20%28m%28x_1%29%2C...%2Cm%28x_N%29%29 "fig:")
-   ![equation](https://latex.codecogs.com/gif.latex?K_%7Bij%7D%3D%5Ckappa%28x_i%2Cx_j%29 "fig:")

![equation](https://latex.codecogs.com/gif.latex?m) is the mean function
and it is common to use
![equation](https://latex.codecogs.com/gif.latex?m%28x%29%20%3D%200) as
GPs are flexible enough to model the mean arbitrarily well.
![equation](https://latex.codecogs.com/gif.latex?%5Ckappa) is a positive
definite *kernel function* or *covariance function*. Thus, a Gaussian
process is a distribution over functions whose shape (smoothness, …) is
defined by ![equation](https://latex.codecogs.com/gif.latex?K). If
points ![equation](https://latex.codecogs.com/gif.latex?x_i) and
![equation](https://latex.codecogs.com/gif.latex?x_j) are considered to
be similar by the kernel the function values at these points,
![equation](https://latex.codecogs.com/gif.latex?f%28x_i%29) and
![equation](https://latex.codecogs.com/gif.latex?f%28x_j%29), can be
expected to be similar too.

Given a training dataset with noise-free function values
![equation](https://latex.codecogs.com/gif.latex?f) at inputs
![equation](https://latex.codecogs.com/gif.latex?X), a GP prior can be
converted into a GP posterior
![equation](https://latex.codecogs.com/gif.latex?p%28f_*%20%7C%20X_*%2C%20X%2C%20f%29)
which can then be used to make predictions
![equation](https://latex.codecogs.com/gif.latex?f_*) at new inputs
![equation](https://latex.codecogs.com/gif.latex?X_*). By definition of
a GP, the joint distribution of observed values
![equation](https://latex.codecogs.com/gif.latex?f) and prediction
![equation](https://latex.codecogs.com/gif.latex?f_*) is again a
Gaussian which can be partitioned into

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%20f%20%5C%5C%20f_*%20%5Cend%7Bpmatrix%7D%20%5Csim%20%5Cmathcal%7BN%7D%20%5Cbegin%7Bpmatrix%7D%200%2C%20%5Cbegin%7Bpmatrix%7D%20K%20%26%20K_*%20%5C%5C%20K_*%5ET%20%26%20K_%7B**%7D%20%5Cend%7Bpmatrix%7D%20%5Cend%7Bpmatrix%7D)

Where,

-   ![equation](https://latex.codecogs.com/gif.latex?K_*%20%3D%20%5Ckappa%28X%2C%20X_*%29 "fig:")
-   ![equation](https://latex.codecogs.com/gif.latex?K_%7B**%7D%20%3D%20%5Ckappa%28X_*%2C%20X_*%29 "fig:")

With ![equation](https://latex.codecogs.com/gif.latex?N) training data
and ![equation](https://latex.codecogs.com/gif.latex?N_*) new input
data,

-   ![equation](https://latex.codecogs.com/gif.latex?K) is a
    ![equation](https://latex.codecogs.com/gif.latex?N%20%5Ctimes%20N)
    matrix
-   ![equation](https://latex.codecogs.com/gif.latex?K_*) is a
    ![equation](https://latex.codecogs.com/gif.latex?N%20%5Ctimes%20N_*)
    matrix
-   ![equation](https://latex.codecogs.com/gif.latex?K_%7B**%7D) is a
    ![equation](https://latex.codecogs.com/gif.latex?N_*%20%5Ctimes%20N_*)
    matrix

Using standard rules for conditioning Gaussians, the predictive
distribution is given by

![equation](https://latex.codecogs.com/gif.latex?p%28f_*%7CX_*%2CX%2Cf%29%20%3D%20%5Cmathcal%7BN%7D%28f_*%7C%5Cmu_*%2C%5CSigma_*%29)

Where,

-   ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_*%20%3D%20K_*%5ET%20K%5E%7B-1%7D%20f "fig:")
-   ![equation](https://latex.codecogs.com/gif.latex?%5CSigma_*%20%3D%20K_%7B**%7D%20-%20K_*%5ET%20K%5E%7B-1%7D%20K_* "fig:")

If we have a training dataset with noisy function values
![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20f%20+%20%5Cepsilon)
where noise
![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon%20%5Csim%20%5Cmathcal%7BN%7D%280%2C%20%5Csigma_y%5E2%20I%29)
is independently added to each observation then the predictive
distribution is given by

![equation](https://latex.codecogs.com/gif.latex?p%28f_*%7CX_*%2CX%2Cy%29%20%3D%20%5Cmathcal%7BN%7D%28f_*%7C%5Cmu_*%2C%5CSigma_*%29)

Where,

-   ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_*%20%3D%20K_*%5ET%20K_y%5E%7B-1%7D%20y "fig:")
-   ![equation](https://latex.codecogs.com/gif.latex?%5CSigma_*%20%3D%20K_%7B**%7D%20-%20K_*%5ET%20K_y%5E%7B-1%7D%20K_* "fig:")
-   ![equation](https://latex.codecogs.com/gif.latex?K_y%20%3D%20K%20+%20%5Csigma_y%5E2I "fig:")

Although Equation (6) covers noise in training data, it is still a
distribution over noise-free predictions
![equation](https://latex.codecogs.com/gif.latex?f_*). To additionally
include noise
![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon) into
predictions ![equation](https://latex.codecogs.com/gif.latex?y_*) we
have to add
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_y%5E2) to the
diagonal of ![equation](https://latex.codecogs.com/gif.latex?%5CSigma_*)

![equation](https://latex.codecogs.com/gif.latex?p%28y_*%7CX_*%2CX%2Cy%29%20%3D%20%5Cmathcal%7BN%7D%28y_*%7C%5Cmu_*%2C%5CSigma_*%20+%20%5Csigma_y%5E2I%29)

using the definitions of
![equation](https://latex.codecogs.com/gif.latex?%5Cmu_*) and
![equation](https://latex.codecogs.com/gif.latex?%5CSigma_*) from
Equations (7) and (8), respectively. This is the minimum we need to know
for implementing Gaussian processes and applying them to regression
problems. For further details, please consult the literature in the
References section. The next section shows how to implement GPs from
scratch.

## Implementation

Here, we will use the squared exponential kernel, also known as Gaussian
kernel or RBF kernel:

![equation](https://latex.codecogs.com/gif.latex?%5Ckappa%28x_i%2C%20x_j%29%20%3D%20%5Csigma_f%5E2%20%5Cexp%20%5Cleft%28%5Cfrac%7B-%7Cx_i%20-%20x_j%7C%5E2%7D%7B2l%5E2%7D%5Cright%29)

The length parameter ![equation](https://latex.codecogs.com/gif.latex?l)
controls the smoothness of the function and
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_f) the
vertical variation. For simplicity, we use the same length parameter
![equation](https://latex.codecogs.com/gif.latex?l) for all input
dimensions (isotropic kernel).

``` r
gaussian_kernel <- function(x, y, l = 1.0, sigma_f = 1.0) {
    if (is.null(dim(x))) {
        sqdist <- outer(x^2, y^2, '+') - 2 * x %*% t(y)
    } else {
        sqdist <- outer(rowSums(x^2), rowSums(y^2), '+') - 2 * x %*% t(y)
    }
    sigma_f^2 * exp(-0.5 / l^2 * sqdist)
}
```

There are many other kernels that can be used for Gaussian processes.
See \[3\] for a detailed reference or the scikit-learn documentation for
some examples.

### Prior

Let’s first define a prior over functions with mean zero and a
covariance matrix computed with kernel parameters
![equation](https://latex.codecogs.com/gif.latex?l%20%3D%201) and
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_f%20%3D%201).
To draw random functions from that GP we draw random samples from the
corresponding multivariate normal. The following example draws three
random samples and plots it together with the zero mean and the 95%
confidence interval (computed from the diagonal of the covariance
matrix).

``` r
# Finite number of points
X <- seq(-5, 5, 0.2)

# Mean and covariance of the prior
mu <- numeric(length(X))
cov <- gaussian_kernel(X, X)

# Draw three samples from the prior
samples <- mvrnorm(n = 3, mu = mu, Sigma = cov)
```

``` r
plot_gp <- function(mu, cov, X, X_train = NULL, Y_train = NULL, samples = NULL) {
    
    # 95% of the area under a gaussian lies within
    # 1.96 standard deviation of the mean.
    # The diagonal of the covariance matrix is the
    # variances of each individual gaussian.
    uncertainty = 1.96 * sqrt(diag(cov))
    
    g <- ggplot(data = data.frame(X = X, Y = mu), aes(X, Y))
    g <- g + geom_line(y = mu, size = 0.7, color = "blue")
    
    if (!is.null(samples)) {
        for (row in 1:nrow(samples)) {
            g <- g + geom_line(y = samples[row,], color = row, linetype = "dashed")
        }
    }
    
    if (!is.null(X_train)) {
        g <- g + geom_point(data = data.frame(X = X_train), y = Y_train,
                            size = 2)
    }
    
    g <- g + geom_ribbon(aes(ymin = mu - uncertainty, ymax = mu + uncertainty),
                         fill = "skyblue", alpha = 0.3, color = "white")
    g <- g + theme_minimal()
    g
}

plot_gp(mu, cov, X, samples = samples)
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

### Prediction from noise-free training data

To compute the sufficient statistics i.e. mean and covariance of the
posterior predictive distribution we implement Equations (4) and (5)

``` r
len <- function(x) {
    if (is.null(dim(x))) {
        length(x)
    } else {
        dim(x)[1]
    }
}

posterior_predictive <- function(X_s, X_train, Y_train, kernel, l=1.0,
                                 sigma_f=1.0, sigma_y=1e-8) {
    
    K <- kernel(X_train, X_train, l, sigma_f) + sigma_y^2 * diag(len(X_train))
    K_s <- kernel(X_train, X_s, l, sigma_f)
    K_ss <- kernel(X_s, X_s, l, sigma_f) + 1e-8 * diag(len(X_s))
    K_inv <- solve(K) #O(n^3)
    
    # Equation (4)
    mu_s <- t(K_s) %*% K_inv %*% Y_train
    
    # Equation (5)
    cov_s <- K_ss - t(K_s) %*% K_inv %*% K_s
    
    result <- list("mu_s" = mu_s, "cov_s" = cov_s)
    return(result)
}
```

and apply them to noise-free training data `X_train` and `Y_train`. The
following example draws three samples from the posterior predictive and
plots them along with mean, confidence interval and training data. In a
noise-free model, variance at the training points is zero and all random
functions drawn from the posterior go through the training points.

``` r
# Noise free training data
X_train <- c(-4,-3,-2,-1,1)
Y_train <- sin(X_train)

# Compute mean and covariance of the posterior predictive distribution
result <- posterior_predictive(X, X_train, Y_train, gaussian_kernel)
mu_s <- result$mu_s
cov_s <- result$cov_s

samples <- mvrnorm(n = 3, mu = mu_s, Sigma = cov_s)
plot_gp(mu_s, cov_s, X, X_train, Y_train, samples)
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### Prediction from noisy training data

If some noise is included in the model, training points are only
approximated and the variance at the training points is non-zero.

``` r
noise <- 0.4

# Noisy training data
X_train <- seq(-3, 3, 1)
Y_train <- sin(X_train) + noise * rnorm(n = length(X_train))

# Compute mean and covariance of the posterior predictive distribution
result <- posterior_predictive(X, X_train, Y_train, gaussian_kernel, sigma_y=noise)
mu_s <- result$mu_s
cov_s <- result$cov_s

samples <- mvrnorm(n = 3, mu = mu_s, Sigma = cov_s)
plot_gp(mu_s, cov_s, X, X_train, Y_train, samples)
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

### Effect of kernel parameters and noise parameter

The following example shows the effect of kernel parameters
![equation](https://latex.codecogs.com/gif.latex?l) and
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_f) as well as
the noise parameter
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_f). Higher
![equation](https://latex.codecogs.com/gif.latex?l) values lead to
smoother functions and therefore to coarser approximations of the
training data. Lower ![equation](https://latex.codecogs.com/gif.latex?l)
values make functions more wiggly with wide confidence intervals between
training data points.
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_f) controls
the vertical variation of functions drawn from the GP. This can be seen
by the wide confidence intervals outside the training data region in the
right figure of the second row.
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_y) represents
the amount of noise in the training data. Higher
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_y) values make
more coarse approximations which avoids overfitting to noisy data.

``` r
params = as.matrix(
    data.frame(
        c(0.3, 3.0, 1.0, 1.0, 1.0, 1.0),
        c(1.0, 1.0, 0.3, 3.0, 1.0, 1.0),
        c(0.2, 0.2, 0.2, 0.2, 0.05, 1.5)
    )
)

plot_list <- list()
for (i in 1:nrow(params)) {
    l <- params[i, 1]
    sigma_f <- params[i, 2]
    sigma_y <- params[i, 3]
    
    result <- posterior_predictive(X, X_train, Y_train, gaussian_kernel,
                                   l=l, sigma_f=sigma_f, sigma_y=sigma_y)
    mu_s <- result$mu_s
    cov_s <- result$cov_s
    
    g <- plot_gp(mu_s, cov_s, X, X_train, Y_train)
    g <- g + ggtitle(paste0("l = ", l, ", sigma_f = ", sigma_f, ", sigma_y = ", sigma_y))
    plot_list[[i]] <- g
}

do.call('grid.arrange', c(plot_list, ncol = 2))
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Optimal values for these parameters can be estimated by maximizing the
marginal log-likelihood which is given by

![equation](https://latex.codecogs.com/gif.latex?%5Clog%20p%28y%7CX%29%20%3D%20%5Clog%20%5Cmathcal%7BN%7D%28y%7C0%2CK_y%29%20%3D%20-%5Cfrac%7B1%7D%7B2%7Dy%5ET%20K_y%5E%7B-1%7D%20y%20-%20%5Cfrac%7B1%7D%7B2%7D%5Clog%7CK_y%7C%20-%20%5Cfrac%7BN%7D%7B2%7D%5Clog%282%20%5Cpi%29)

In the following we will minimize the negative marginal log-likelihood
w.r.t parameters ![equation](https://latex.codecogs.com/gif.latex?l) and
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_f).
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_y) is set to
the known noise level of the data. If the noise level is unknown,
![equation](https://latex.codecogs.com/gif.latex?%5Csigma_y) can be
estimated as well along with the other parameters.

``` r
nll_fn <- function(X_train, Y_train, noise, kernel) {
    step <- function(theta) {
        K <- kernel(X_train, X_train, l=theta[1], sigma_f=theta[2]) +
             noise^2 * diag(len(X_train))
        
        # Compute determinant via Cholesky decomposition
        # log(det(A)) = 2 * sum(log(diag(L)))
        return(sum(log(diag(chol(K)))) +
               0.5 * t(Y_train) %*% solve(K) %*% Y_train +
               0.5 * len(X_train) * log(2 * pi))
    }
    
    return(step)
}

# Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
# We should actually run the minimization several times with different
# initializations to avoid local minima but this is skipped here for
# simplicity
(res <- optim(par = c(1, 1),
             fn = nll_fn(X_train, Y_train, noise, gaussian_kernel),
             lower = rep(1e-5, 2),
             upper = rep(Inf, 2),
             method = "L-BFGS-B"))
```

    ## $par
    ## [1] 1.1624787 0.4252034
    ## 
    ## $value
    ## [1] 4.930078
    ## 
    ## $counts
    ## function gradient 
    ##        9        9 
    ## 
    ## $convergence
    ## [1] 0
    ## 
    ## $message
    ## [1] "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH"

``` r
# Store the optimization results in global variables so that we can
# compare it later with the results from other implementations.
l_opt <- res$par[1]
sigma_f_opt <- res$par[2]

# Compute the posterior predictive statistics with optimized kernel parameters and plot the results
result <- posterior_predictive(X, X_train, Y_train, gaussian_kernel, l=l_opt,
                               sigma_f=sigma_f_opt, sigma_y=noise)
plot_gp(mu_s, cov_s, X, X_train, Y_train)
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

With optimized kernel parameters, training data are reasonably covered
by the 95% confidence interval and the mean of the posterior predictive
is a good approximation.

### Higher dimensions

The above implementation can also be used for higher input data
dimensions. Here, a GP is used to fit noisy samples from a sine wave
originating at 0 and expanding in the x-y plane. The following plots
show the noisy samples and the posterior predictive mean before and
after kernel parameter optimization.

``` r
noise_2D <- 0.1

rx <- seq(-5, 5, 0.3)

X_2D <- as.matrix(
    data.frame(
        rep(rx, times = length(rx)),
        rep(rx, each = length(rx))
    )
)

X_2D_train <- as.matrix(
    data.frame(
        runif(n = 100, min = -4, max = 4),
        runif(n = 100, min = -4, max = 4)
    )
)

Y_2D_train <- sin(0.5 * sqrt(X_2D_train[,1]^2 + X_2D_train[,2]^2)) +
              noise_2D * rnorm(len(X_2D_train))

result <- posterior_predictive(X_2D, X_2D_train, Y_2D_train, gaussian_kernel, sigma_y=noise_2D)
mu_s <- array(result$mu_s, c(length(rx), length(rx)))

persp3D(x=rx, y=rx, z=mu_s,
        theta=30, phi=20, alpha=0.5)
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
res <- optim(par = c(1, 1),
             fn = nll_fn(X_2D_train, Y_2D_train, noise_2D, gaussian_kernel),
             lower = rep(1e-5, 2),
             upper = rep(Inf, 2),
             method = "L-BFGS-B")
l_2D_opt <- res$par[1]
sigma_f_2D_opt <- res$par[2]

result <- posterior_predictive(X_2D, X_2D_train, Y_2D_train, gaussian_kernel,
                               l=l_2D_opt, sigma_f=sigma_f_2D_opt, sigma_y=noise_2D)
mu_s <- array(result$mu_s, c(length(rx), length(rx)))

persp3D(x=rx, y=rx, z=mu_s,
        theta=30, phi=20, alpha=0.5)
```

![](01_gaussian-process_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->
