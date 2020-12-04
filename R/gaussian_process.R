len <- function(x) {
  if (is.null(dim(x))) {
    length(x)
  } else {
    dim(x)[1]
  }
}

rappend <- function(X, x) {
  if (is.null(X)) {
    X <- x
  } else if (is.null(dim(X)) & length(x) <= 1) {
    append(X, x)
  } else {
    rbind(X, x)
  }
}

#' Title
#'
#' @param x 
#' @param y 
#' @param l 
#' @param sigma_f 
#'
#' @return
#' @export
#'
#' @examples
gaussian_kernel <- function(x, y, l = 1.0, sigma_f = 1.0) {
  if (is.null(dim(x))) {
    sqdist <- outer(x^2, y^2, '+') - 2 * x %*% t(y)
  } else {
    sqdist <- outer(rowSums(x^2), rowSums(y^2), '+') - 2 * x %*% t(y)
  }
  sigma_f^2 * exp(-0.5 / l^2 * sqdist)
}

#' Title
#'
#' @param mu 
#' @param cov 
#' @param X 
#' @param X_train 
#' @param Y_train 
#' @param samples 
#'
#' @return
#' @export
#'
#' @examples
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

#' Title
#'
#' @param X_train 
#' @param Y_train 
#' @param noise 
#' @param kernel 
#'
#' @return
#' @export
#'
#' @examples
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

#' Title
#'
#' @param l 
#' @param sigma_f 
#' @param sigma_y 
#' @param kern 
#'
#' @return
#' @export
#'
#' @examples
gpr.init <- function(l=1.0, sigma_f=1.0, sigma_y=0, kern=gaussian_kernel) {
  list("X_train" = NULL,
       "Y_train" = NULL,
       "l" = l,
       "sigma_f" = sigma_f,
       "sigma_y" = sigma_y,
       "kernel" = gaussian_kernel
       )
}

#' Title
#'
#' @param X_train 
#' @param Y_train 
#' @param gpr 
#' @param lower 
#' @param upper 
#' @param n_restarts 
#'
#' @return
#' @export
#'
#' @examples
gpr.fit <- function(X_train, Y_train, gpr, lower=c(1e-5, 1e-5),
                    upper=c(9, 9), n_restarts=25) {
  
  gpr$X_train <- X_train
  gpr$Y_train <- Y_train
  
  min_val <- Inf
  min_x <- NULL
  
  for (i in 1:n_restarts) {
    x0 <- runif(n = len(lower), min = lower, max = upper)
    res <- optim(par = x0, fn = nll_fn(X_train, Y_train, gpr$sigma_y, gpr$kernel),
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

#' Title
#'
#' @param X 
#' @param gpr 
#'
#' @return
#' @export
#'
#' @examples
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