#' Title
#'
#' @param X 
#' @param X_sample 
#' @param Y_sample 
#' @param gpr 
#' @param xi 
#' @param max 
#'
#' @return
#' @export
#'
#' @examples
expected_improvement <- function(X, X_sample, Y_sample, gpr, xi=0.01, max=TRUE) {
  result <- gpr.predict(X, gpr)
  mu <- result$mu_s
  sigma <- diag(result$cov_s)
  
  result <- gpr.predict(X_sample, gpr)
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

#' Title
#'
#' @param acquisition 
#' @param X_sample 
#' @param Y_sample 
#' @param gpr 
#' @param lower 
#' @param upper 
#' @param n_restarts 
#' @param max 
#' @param xi 
#'
#' @return
#' @export
#'
#' @examples
propose_location <- function(acquisition, X_sample, Y_sample, gpr,
                             lower, upper, n_restarts=25, xi=0.01, max=TRUE) {
  
  min_obj <- function(X) {
    if (length(X) > 1) {
      X <- t(X)
    }
    # Minimization objective is the negative acquisition function
    -acquisition(X, X_sample, Y_sample, gpr, xi=xi, max=max)
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


#' Title
#'
#' @param FUN 
#' @param lower 
#' @param upper 
#' @param init_grid_dt 
#' @param init_points 
#' @param n_iter 
#' @param xi 
#' @param noise 
#' @param max 
#' @param acq 
#'
#' @return
#' @export
#'
#' @examples
bayesian_optimization <- function(FUN, lower, upper, init_grid_dt=NULL, init_points=1,
                                  n_iter=10, xi=0.01, noise=0, max=TRUE, acq=expected_improvement) {
  
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
    gpr <- gpr.fit(X_train, Y_train, gpr)
    
    # Obtain next sampling point from the acquisition function
    X_next <- propose_location(acq, X_train, Y_train, gpr, lower, upper, xi=xi, max=max)
    
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

#' Title
#'
#' @param X 
#' @param EI 
#' @param X_next 
#'
#' @return
#' @export
#'
#' @examples
plot_acquisition <- function(X, EI, X_next) {
  ggplot(data=data.frame(x = X, y = EI), aes(x, y)) +
    geom_line(color = "red", size = 0.7) +
    geom_vline(xintercept = X_next, linetype = "dashed") +
    theme_minimal()
}