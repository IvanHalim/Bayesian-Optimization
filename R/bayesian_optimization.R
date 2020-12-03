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
  if (max) {
    mu_sample_opt <- max(mu_sample)
  } else {
    mu_sample_opt <- min(mu_sample)
  }
  
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
#'
#' @return
#' @export
#'
#' @examples
propose_location <- function(acquisition, X_sample, Y_sample, gpr,
                             lower, upper, n_restarts=25, xi=0.01, max=TRUE) {
  
  min_obj <- function(X) {
    # Minimization objective is the negative acquisition function
    -acquisition(t(X), X_sample, Y_sample, gpr, xi=xi, max=max)
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