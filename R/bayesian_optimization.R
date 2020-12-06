#' Expected Improvement Acquisition Function
#' 
#' Computes the EI at points X based on existing samples X_sample
#' and Y_sample using a Gaussian process surrogate model
#'
#' @param X points at which EI shall be computed `m * d`.
#' @param gpr a fitted gaussian process object containing `X_train` and `Y_train`.
#' @param xi exploitation-exploration trade-off parameter
#' @param max if TRUE we want to maximize the fitness function, otherwise minimize.
#'
#' @return
#' A numeric vector containing the expected improvements at points `X`
#' 
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
#' 
#' # Finite number of points
#' X <- seq(-5, 5, 0.2)
#' EI <- expected_improvement(X, gpr)
expected_improvement <- function(X, gpr, xi=0.01, max=TRUE) {
  result <- gpr.predict(X, gpr)
  mu <- result$mu_s
  sigma <- diag(result$cov_s)
  
  result <- gpr.predict(gpr$X_train, gpr)
  mu_sample <- result$mu_s
  
  # Needed for noise-based model
  # otherwise use max(Y_sample)
  mu_sample_opt <- ifelse(max, max(mu_sample), min(mu_sample))

  imp <- mu - mu_sample_opt - xi
  Z <- imp / sigma
  expected_imp <- imp * pnorm(Z) + sigma * dnorm(Z)
  expected_imp[sigma == 0] = 0
  
  return(expected_imp)
}

#' Propose Sampling Location
#' 
#' Proposes the next sampling point by optimizing the acquisition function.
#'
#' @param acquisition The acquisition function to be used.
#' @param gpr a fitted gaussian process object containing `X_train` and `Y_train`.
#' @param lower the lower bounds of each variable in `X`.
#' @param upper the upper bounds of each variable in `X`.
#' @param n_restarts the number of iterations to find the optimum acquisition.
#' @param max if TRUE we want to maximize the fitness function, otherwise minimize.
#' @param xi exploitation-exploration trade-off parameter.
#'
#' @return Location of the acquisition function maximum.
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
#' 
#' # Obtain next sampling point from the acquisition function
#' lower <- c(-5, -5)
#' upper <- c(5, 5)
#' X_next <- propose_location(expected_improvement, gpr, lower, upper)
propose_location <- function(acquisition, gpr, lower, upper,
                             n_restarts=25, xi=0.01, max=TRUE) {
  
  min_obj <- function(X) {
    if (length(X) > 1) {
      X <- t(X)
    }
    # Minimization objective is the negative acquisition function
    -acquisition(X, gpr, xi=xi, max=max)
  }
  
  min_val = 1
  min_x = NULL
  
  # Find the best optimum by starting from n_restart different random points.
  for (i in 1:n_restarts) {
    x0 <- runif(n = len(lower), min = lower, max = upper)
    res <- optim(par = x0, fn = min_obj, lower = lower,
                 upper = upper, method = "L-BFGS-B")
    
    if (res$value < min_val) {
      min_val <- res$value
      min_x <- res$par
    }
  }
  
  return(min_x)
}


#' Bayesian Optimization
#' 
#' Perform a Bayesian Optimization on a fitness function.
#'
#' @param FUN the fitness function.
#' @param lower the lower bounds of each variables.
#' @param upper the upper bounds of each variables.
#' @param init_grid_dt user specified points to sample the target function.
#' @param init_points number of randomly chosen points to sample the target function before Bayesian Optimization fitting the Gaussian Process.
#' @param n_iter number of repeated Bayesian Optimization.
#' @param xi exploitation-exploration trade-off parameter.
#' @param noise represents the amount of noise in the training data.
#' @param max if TRUE we want to maximize the fitness function, otherwise minimize.
#' @param acq choice of acquisition function (Expected Improvement by default).
#' @param naive if TRUE use the naive a naive implementation of negative log-likelihood,
#' otherwise use a numerically more stable implementation.
#'
#' @return a list with components
#' \itemize{
#'    \item `par` the best set of parameters found.
#'    \item `value` the value of `FUN` corresponding to `par`
#' }
#' @export
#'
#' @examples
#' noise <- 0.2 
#' f <- function(X, noise = noise) {
#'   -sin(3*X) - X^2 + 0.7*X + noise * rnorm(length(X))
#' }
#' 
#' search_grid <- c(-0.9, 1.1)
#' lower <- -1.0
#' upper <- 2.0
#' 
#' (result <- bayesian_optimization(FUN=f, lower=lower, upper=upper, init_grid_dt=search_grid,
#'                                  noise=noise, naive=TRUE))
bayesian_optimization <- function(FUN, lower, upper, init_grid_dt=NULL, init_points=0,
                                  n_iter=10, xi=0.01, noise=0, max=TRUE, acq=expected_improvement, naive=FALSE) {
  
  X_train <- init_grid_dt
  
  if (init_points > 0) {
    for (i in 1:init_points) {
      X <- runif(length(lower), lower, upper)
      rappend(X_train, X)
    }
  }
  
  Y_train <- FUN(X_train, noise)
  
  gpr <- gpr.init(sigma_y=noise)
  
  for (i in 1:n_iter) {
    # Update Gaussian process with existing samples
    gpr <- gpr.fit(X_train, Y_train, gpr, naive=naive)
    
    # Obtain next sampling point from the acquisition function
    X_next <- propose_location(acq, gpr, lower, upper, xi=xi, max=max)
    
    # Obtain next noisy sample from the objective function
    Y_next <- FUN(X_next, noise)
    
    # Add sample to previous samples
    X_train <- rappend(X_train, X_next)
    Y_train <- rappend(Y_train, Y_next)
  }
  
  best_val <- ifelse(max, max(Y_train), min(Y_train))
  best_data <-  filter(data.frame(X_train, Y_train), Y_train == best_val)
  
  par <- unlist(best_data)[-ncol(best_data)]
  
  list("par" = par, "value" = best_val)
}

#' Plotting Acquisition Function
#' 
#' A utility function to plot the acquisition function of a gaussian process
#'
#' @param X the `x` coordinates of points in the plot.
#' @param acq the `y` coordinates of points in the plot, which are the acquisition values of `x`.
#' @param X_next the proposed next location to be sampled from.
#'
#' @return a ggplot object
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
#' 
#' # Finite number of points
#' X <- seq(-5, 5, 0.2)
#' EI <- expected_improvement(X, gpr)
#' plot_acquisition(X, EI)
plot_acquisition <- function(X, acq, X_next=NULL) {
  g <- ggplot(data=data.frame(x = X, y = acq), aes(x, y))
  g <- g + geom_line(color = "red", size = 0.7)
  if (!is.null(X_next)) {
    g <- g + geom_vline(xintercept = X_next, linetype = "dashed")
  }
  g <- g + theme_minimal()
  g
}