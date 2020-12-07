#' Generalized length function
#' 
#' Returns the length of `x` if it's a vector or the number of rows if `x` is an array.
#' 
#' @param x a vector or a two-dimensional array.
#'
#' @return an integer of length 1 or NULL.
#' @export
#'
#' @examples
#' len(c(1,2,3))
#' len(diag(4))
len <- function(x) {
  if (is.null(dim(x))) {
    length(x)
  } else {
    dim(x)[1]
  }
}

#' Generalized append function
#' 
#' Appends a row vector to a matrix or an element to a vector.
#' In the case of two vectors, stack them on top of each other
#' to form a two-dimensional matrix.
#'
#' @param X the vector or two-dimensional matrix to be appended to
#' @param Y the row vector or numeric value to be added
#'
#' @return
#' A vector containing the values in `X` with `Y` appended
#' at the end of the vector or a matrix `X` with the row vector
#' `Y` added to the bottom.
#' 
#' @export
#'
#' @examples
#' rappend(c(1,2,3), 4)
#' rappend(diag(3), c(1,2,3))
#' rappend(c(1,2,3), c(4,5,6))
rappend <- function(X, Y) {
  if (is.null(X)) {
    X <- Y
  } else if (is.null(dim(X)) & length(Y) <= 1) {
    append(X, Y)
  } else {
    rbind(X, Y)
  }
}

#' Isotropic RBF Kernel
#' 
#' Compute the squared euclidean distance similarity measure between two sets of points.
#'
#' @param x a numeric vector or a two-dimensional array of length `m`.
#' @param y a numeric vector or a two-dimensional array of length `n`. Must be the same type of object as `x`.
#' @param l the length parameter that controls the smoothness of the function.
#' @param sigma_f the scale parameter that controls the vertical variation of the function.
#'
#' @return an `m * n` similarity matrix
#' @export
#'
#' @examples
#' gaussian_kernel(c(1,2,3,4), c(5,6))
#' gaussian_kernel(diag(3), t(c(1,2,3))
gaussian_kernel <- function(x, y, l = 1.0, sigma_f = 1.0) {
  if (is.null(dim(x))) {
    sqdist <- outer(x^2, y^2, '+') - 2 * x %*% t(y)
  } else {
    sqdist <- outer(rowSums(x^2), rowSums(y^2), '+') - 2 * x %*% t(y)
  }
  sigma_f^2 * exp(-0.5 / l^2 * sqdist)
}

#' Plotting Gaussian Process
#'
#' A utility function to plot a Gaussian Process
#'
#' @param mu a mean vector of length `m`.
#' @param cov an `m * m` covariance matrix.
#' @param the `x` coordinates of points in the plot.
#' @param X_train the `x` coordinates of the training data.
#' @param Y_train the `y` coordinates of the training data.
#' @param samples drawn from a multivariate normal distribution,
#' an `n * length(mu)` matrix with one sample in each row`
#'
#' @return a ggplot object
#' @export
#'
#' @examples
#' noise <- 0.4
#' gpr <- gpr.init(sigma_y = noise)
#' 
#' # Finite number of points
#' X <- seq(-5, 5, 0.2)
#' 
#' # Noisy training data
#' X_train <- seq(-3, 3, 1)
#' Y_train <- sin(X_train) + noise * rnorm(n = length(X_train))
#' gpr <- gpr.fit(X_train, Y_train, gpr)
#' 
#' # Compute mean and covariance of the posterior distribution
#' result <- gpr.predict(X, gpr)
#' mu_s <- result$mu_s
#' cov_x <- result$cov_s
#' 
#' samples <- mvrnorm(n = 3, mu = mu_s, Sigma = cov_s)
#' plot_gp(mu_s, cov_s, X, X_train, Y_train, samples)
plot_gp <- function(mu, cov, X, X_train = NULL, Y_train = NULL, samples = NULL) {
  
  # 95% of the area under a gaussian lies within
  # 1.96 standard deviation of the mean.
  # The diagonal of the covariance matrix is the
  # variances of each individual gaussian.
  uncertainty = 1.96 * sqrt(diag(cov))
  
  g <- ggplot2::ggplot(data = data.frame(X = X, Y = mu), aes(X, Y))
  g <- g + ggplot2::geom_line(y = mu, size = 0.7, color = "blue")
  
  if (!is.null(samples)) {
    for (row in 1:nrow(samples)) {
      g <- g + ggplot2::geom_line(y = samples[row,], color = row, linetype = "dashed")
    }
  }
  
  if (!is.null(X_train)) {
    g <- g + ggplot2::geom_point(data = data.frame(X = X_train), y = Y_train,
                        size = 2)
  }
  
  g <- g + ggplot2::geom_ribbon(aes(ymin = mu - uncertainty, ymax = mu + uncertainty),
                       fill = "skyblue", alpha = 0.3, color = "white")
  g <- g + ggplot2::theme_minimal()
  g
}

#' Negative log-likelihood function
#' 
#' Returns a function that computes the negative log marginal
#' likelihood for training data `X_train` and `Y_train` and given
#' noise level.
#'
#' @param X_train training location `m * d`
#' @param Y_train training targets `m * 1`
#' @param noise known noise level of `Y_train`
#' @param kernel kernel function used
#' @param naive if TRUE use a naive implementation, if FALSE use a numerically more stable implementation.
#'
#' @return
#' A function that computes the negative log-likelihood w.r.t.
#' parameters `l` and `sigma_f`
#' @export
#'
#' @examples
#' noise <- 0.4
#' X_train <- seq(-3, 3, 1)
#' Y_train <- sin(X_train) + noise * rnorm(n = length(X_train))
#' nll_fn(X_train, Y_train, noise, gaussian_kernel)
nll_fn <- function(X_train, Y_train, noise, kernel, naive=FALSE) {
  
  nll_naive <- function(theta) {
    # Naive implementation of Eq. (11). Works well for the examples 
    # in this article but is numerically less stable compared to 
    # the implementation in nll_stable below
    
    K <- kernel(X_train, X_train, l=theta[1], sigma_f=theta[2]) +
      noise^2 * diag(len(X_train))
    # Compute determinant via Cholesky decomposition
    # log(det(A)) = 2 * sum(log(diag(l)))
    return(sum(log(diag(chol(K)))) +
             0.5 * t(Y_train) %*% solve(K) %*% Y_train +
             0.5 * len(X_train) * log(2 * pi))
  }
  
  nll_stable <- function(theta) {
    # Numerically more stable implementation of Eq. (11) as described
    # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
    # 2.2, Algorithm 2.1
    
    ls <- function(X, Y) {
      # Compute least squares using QR decomposition
      QR <- qr(X)
      Q <- qr.Q(QR)
      R <- qr.R(QR)
      solve(R) %*% Conj(t(Q)) %*% Y
    }
    
    K <- kernel(X_train, X_train, l=theta[1], sigma_f=theta[2]) +
      noise^2 * diag(len(X_train))
    L <- chol(K)
    return(sum(log(diag(L))) +
             0.5 * Y_train %*% ls(t(L), ls(L, Y_train)) +
             0.5 * len(X_train) * log(2 * pi))
  }
  
  if (naive) {
    return(nll_naive)
  } else {
    return(nll_stable)
  }
}

#' Initialize Gaussian Process
#' 
#' Create a Gaussian Process object to store important parameters
#' for posterior prediction
#'
#' @param l the length parameter that controls the smoothness of the function.
#' @param sigma_f the scale parameter that controls the vertical variation of the function.
#' @param sigma_y the noise parameter, represents the amount of noise in the training data.
#' @param kern the kernel function that we're going to be using.
#'
#' @return a list with all the parameter values as its components
#' @export
#'
#' @examples
#' gpr.init(l=1.0, sigma_f=1.0, sigma_y=0, kern=gaussian_kernel)
gpr.init <- function(l=1.0, sigma_f=1.0, sigma_y=0, kern=gaussian_kernel) {
  list("X_train" = NULL,
       "Y_train" = NULL,
       "l" = l,
       "sigma_f" = sigma_f,
       "sigma_y" = sigma_y,
       "kernel" = gaussian_kernel
       )
}

#' Fitting Gaussian Process
#' 
#' Trains the Gaussian Process model based on training data. It then
#' finds the optimum values for parameters `l` and `sigma_f`
#'
#' @param X_train training location `m * d`
#' @param Y_train training targets `m * 1`
#' @param gpr a gaussian process object
#' @param lower lower bound of `l` and `sigma_f`
#' @param upper upper bound of `l` and `sigma_f`
#' @param n_restarts number of iterations to find the optimum `l` and `sigma_f`
#' @param naive if TRUE use a naive implementation, if FALSE use a numerically more stable implementation.
#'
#' @return
#' an updated gaussian process object which contains `X_train`, `Y_train`, `l` and `sigma_f`
#' @export
#'
#' @examples
#' noise <- 0.4
#' gpr <- gpr.init(sigma_y = noise)
#' 
#' # Noisy training data
#' X_train <- seq(-3, 3, 1)
#' Y_train <- sin(X_train) + noise * rnorm(n = length(X_train))
#' gpr <- gpr.fit(X_train, Y_train, gpr)
gpr.fit <- function(X_train, Y_train, gpr, lower=c(1e-5, 1e-5),
                    upper=c(2, 2), n_restarts=25, naive=FALSE) {
  
  gpr$X_train <- X_train
  gpr$Y_train <- Y_train
  
  min_val <- Inf
  min_x <- NULL
  
  for (i in 1:n_restarts) {
    x0 <- runif(n = len(lower), min = lower, max = upper)
    res <- optim(par = x0, fn = nll_fn(X_train, Y_train, gpr$sigma_y, gpr$kernel, naive),
                 lower = lower, upper = upper, method = "L-BFGS-B")
    
    if (res$value < min_val) {
      min_val <- res$value
      min_x <- res$par
    }
  }
  
  gpr$l <- min_x[1]
  gpr$sigma_f <- min_x[2]
  
  gpr
}

#' Posterior Predictive Distribution
#' 
#' Computes the mean vector and the covariance matrix of the
#' posterior distribution from `m` training data and `n` new inputs.
#'
#' @param X new input locations `n * d`.
#' @param gpr a classifier object containing training data and additional parameters.
#'
#' @return Posterior mean vector `n * d` and covariance matrix `n * n`.
#' @export
#'
#' @examples
#' noise <- 0.4
#' gpr <- gpr.init(sigma_y = noise)
#' 
#' # Finite number of points
#' X <- seq(-5, 5, 0.2)
#' 
#' # Noisy training data
#' X_train <- seq(-3, 3, 1)
#' Y_train <- sin(X_train) + noise * rnorm(n = length(X_train))
#' gpr <- gpr.fit(X_train, Y_train, gpr)
#' 
#' # Compute mean and covariance of the posterior distribution
#' result <- gpr.predict(X, gpr)
#' mu_s <- result$mu_s
#' cov_x <- result$cov_s
gpr.predict <- function(X, gpr) {
  
  X_train <- gpr$X_train
  Y_train <- gpr$Y_train
  kernel <- gpr$kernel
  l <- gpr$l
  sigma_f <- gpr$sigma_f
  sigma_y <- gpr$sigma_y
  
  K <- kernel(X_train, X_train, l, sigma_f) + sigma_y^2 * diag(len(X_train))
  K_s <- kernel(X_train, X, l, sigma_f)
  K_ss <- kernel(X, X, l, sigma_f) + 1e-8 * diag(len(X))
  error_matrix <<- K
  K_inv <- solve(K) #O(n^3)
  
  # Equation (4)
  mu_s <- t(K_s) %*% K_inv %*% Y_train
  
  # Equation (5)
  cov_s <- K_ss - t(K_s) %*% K_inv %*% K_s
  
  result <- list("mu_s" = mu_s, "cov_s" = cov_s)
  return(result)
}

#' Row-Wise Matrix Normalization
#' 
#' Normalize a two-dimensional numeric array such that its row sums add up to 1.
#'
#' @param X a two-dimensional `m * n` numeric array.
#'
#' @return an `m * n` normalized numeric array.
#' @export
#'
#' @examples
#' a <- array(c(1,4,2,5,3,6),c(2,3))
#' normalize(a)
normalize <- function(X) {
  sweep(X, 1, rowSums(X), FUN="/")
}