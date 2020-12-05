Algorithm Evaluation
================
Ivan Timothy Halim
12/3/2020

## Define Parameters

Let’s define the search boundary

``` r
lower <- rep(0, 3)
upper <- rep(1, 3)
```

Let’s set the initial sample

``` r
set.seed(123)
search_grid <- as.matrix(
                  data.frame(
                    w1 = runif(20,0,1),
                    w2 = runif(20,0,1),
                    w3 = runif(20,0,1)
                  )
                )

head(search_grid)
```

    ##             w1        w2        w3
    ## [1,] 0.2875775 0.8895393 0.1428000
    ## [2,] 0.7883051 0.6928034 0.4145463
    ## [3,] 0.4089769 0.6405068 0.4137243
    ## [4,] 0.8830174 0.9942698 0.3688455
    ## [5,] 0.9404673 0.6557058 0.1524447
    ## [6,] 0.0455565 0.7085305 0.1388061

## Run the Algorithm

Let’s run the algorithm `bayesian_optimization()` that we implemented.
The parameters include:

  - `FUN`: the fitness function
  - `lower`: the lower bounds of each variables
  - `upper`: the upper bounds of each variables
  - `init_grid_dt`: user specified points to sample the target function
  - `init_points`: Number of randomly chosen points to sample the target
    function before Bayesian Optimization fitting the Gaussian Process
  - `n_iter`: number of repeated Bayesian Optimization
  - `xi`: tunable parameter \(\xi\) of Expected Improvement, to balance
    exploitation against exploration, increasing `xi` will make the
    optimized hyper parameters more spread out across the whole range
  - `noise`: represents the amount of noise in the training data
  - `max`: specifies whether we’re maximizing or minimizing a function
  - `acq`: choice of acquisition function (Expected Improvement by
    default)

<!-- end list -->

``` r
(bayes_finance <- bayesian_optimization(FUN=sharpe_ratio, lower=lower, upper=upper,
                                        init_grid_dt=search_grid, init_points=10))
```

    ## $par
    ##        w1        w2        w3 
    ## 0.0000000 0.3993422 0.6012231 
    ## 
    ## $value
    ## [1] -304.0334

Result of the function consists of a list with 2 components:

  - par: a vector of the best hyperparameter set found
  - value: the value of metrics achieved by the best hyperparameter set

So, what is the optimum Sharpe Ratio from Bayesian optimization?

``` r
bayes_finance$value
```

    ## [1] -304.0334

The greater a portfolio’s Sharpe ratio, the better its risk-adjusted
performance. If the analysis results in a negative Sharpe ratio, it
either means the risk-free rate is greater than the portfolio’s return,
or the portfolio’s return is expected to be negative.

Let’s check the total weight of the optimum result.

``` r
sum(bayes_finance$par)
```

    ## [1] 1.000565

The sum slightly exceeds 1, which is probably the reason why our
Sharpe’s Ratio is negative. More work can be done to improve sampling
for next \(x\) as well as finding the optimum value for parameters \(l\)
and \(\sigma_f\).

Based on Bayesian Optimization, here is how your asset should be
distributed.

``` r
(bayes_result <- data.frame(stock = unique(nyse$symbol),
                            weight = bayes_finance$par) %>%
                  arrange(desc(weight)) %>%
                  mutate(weight = percent(weight, accuracy = 0.01)) %>%
                  left_join(securities, by = "stock") %>%
                  select(stock, Security, everything()))
```

    ##   stock                             Security weight
    ## 1  ULTA Ulta Salon Cosmetics & Fragrance Inc 60.12%
    ## 2  ORLY                  O'Reilly Automotive 39.93%
    ## 3   NFX              Newfield Exploration Co  0.00%

## `rBayesianOptimization` Package

Let’s compare our result to the one obtained using
`rBayesianOptimization` package.

We need to redefine the fitness function to suit the
`BayesianOptimization` function from `rBayesianOptimization` package.

``` r
fitness <- function(w1, w2, w3) {
    # Assign weight for each stocks
    weight_stock <- c(w1, w2, w3)
    
    # Calculate the numerator
    f1 <- weight_stock * mean_stock$mean
    
    # Calculate the denominator
    f2 <- rowSums(outer(weight_stock, weight_stock) * nyse_cov)
    
    # Calculate Fitness Value
    fitness <- (sum(f1) - rf)/sum(f2)
    
    # Penalize Constraint Violation
    fitness <- fitness - 1e9 * (round(sum(weight_stock), 10) - 1)^2
    
    result <- list(Score = fitness, Pred = 0)
    return(result)
}
```

``` r
search_bound <- list(w1 = c(0,1), w2 = c(0,1),
                     w3 = c(0,1))
search_grid_df <- data.frame(search_grid)

set.seed(1)
rbayes_finance <- BayesianOptimization(FUN = fitness, bounds = search_bound, 
                     init_grid_dt = search_grid_df, init_points = 0, 
                     n_iter = 10, acq = "ei")
```

    ## elapsed = 0.02   Round = 1   w1 = 0.2876 w2 = 0.8895 w3 = 0.1428 Value = -1.023468e+08 
    ## elapsed = 0.00   Round = 2   w1 = 0.7883 w2 = 0.6928 w3 = 0.4145 Value = -8.021977e+08 
    ## elapsed = 0.00   Round = 3   w1 = 0.4090 w2 = 0.6405 w3 = 0.4137 Value = -2.145617e+08 
    ## elapsed = 0.00   Round = 4   w1 = 0.8830 w2 = 0.9943 w3 = 0.3688 Value = -1.552847e+09 
    ## elapsed = 0.00   Round = 5   w1 = 0.9405 w2 = 0.6557 w3 = 0.1524 Value = -5.604287e+08 
    ## elapsed = 0.00   Round = 6   w1 = 0.0456 w2 = 0.7085 w3 = 0.1388 Value = -1.147189e+07 
    ## elapsed = 0.00   Round = 7   w1 = 0.5281 w2 = 0.5441 w3 = 0.2330 Value = -9.315046e+07 
    ## elapsed = 0.00   Round = 8   w1 = 0.8924 w2 = 0.5941 w3 = 0.4660 Value = -9.073010e+08 
    ## elapsed = 0.00   Round = 9   w1 = 0.5514 w2 = 0.2892 w3 = 0.2660 Value = -1.135660e+07 
    ## elapsed = 0.00   Round = 10  w1 = 0.4566 w2 = 0.1471 w3 = 0.8578 Value = -2.130340e+08 
    ## elapsed = 0.00   Round = 11  w1 = 0.9568 w2 = 0.9630 w3 = 0.0458 Value = -9.325547e+08 
    ## elapsed = 0.00   Round = 12  w1 = 0.4533 w2 = 0.9023 w3 = 0.4422 Value = -6.365379e+08 
    ## elapsed = 0.00   Round = 13  w1 = 0.6776 w2 = 0.6907 w3 = 0.7989 Value = -1.362358e+09 
    ## elapsed = 0.00   Round = 14  w1 = 0.5726 w2 = 0.7955 w3 = 0.1219 Value = -2.401001e+08 
    ## elapsed = 0.00   Round = 15  w1 = 0.1029 w2 = 0.0246 w3 = 0.5609 Value = -9.704073e+07 
    ## elapsed = 0.00   Round = 16  w1 = 0.8998 w2 = 0.4778 w3 = 0.2065 Value = -3.412339e+08 
    ## elapsed = 0.00   Round = 17  w1 = 0.2461 w2 = 0.7585 w3 = 0.1275 Value = -1.744483e+07 
    ## elapsed = 0.00   Round = 18  w1 = 0.0421 w2 = 0.2164 w3 = 0.7533 Value = -1.386400e+05 
    ## elapsed = 0.00   Round = 19  w1 = 0.3279 w2 = 0.3182 w3 = 0.8950 Value = -2.928402e+08 
    ## elapsed = 0.00   Round = 20  w1 = 0.9545 w2 = 0.2316 w3 = 0.3745 Value = -3.142636e+08 
    ## elapsed = 0.00   Round = 21  w1 = 0.9749 w2 = 0.0000 w3 = 0.0000 Value = -6.276665e+05 
    ## elapsed = 0.00   Round = 22  w1 = 0.0000 w2 = 0.0000 w3 = 1.0000 Value = 16.2381 
    ## elapsed = 0.00   Round = 23  w1 = 0.7114 w2 = 0.0000 w3 = 0.2841 Value = -2.088616e+04 
    ## elapsed = 0.00   Round = 24  w1 = 0.0000 w2 = 0.9980 w3 = 0.0000 Value = -4032.3122 
    ## elapsed = 0.00   Round = 25  w1 = 0.7790 w2 = 0.1572 w3 = 0.0581 Value = -3.316038e+04 
    ## elapsed = 0.00   Round = 26  w1 = 0.0076 w2 = 0.4095 w3 = 0.5954 Value = -1.575054e+05 
    ## elapsed = 0.00   Round = 27  w1 = 0.0000 w2 = 0.5964 w3 = 0.4066 Value = -9006.5778 
    ## elapsed = 0.00   Round = 28  w1 = 0.0020 w2 = 0.8404 w3 = 0.1630 Value = -2.871837e+04 
    ## elapsed = 0.00   Round = 29  w1 = 0.1507 w2 = 0.0048 w3 = 0.8444 Value = -21.0078 
    ## elapsed = 0.00   Round = 30  w1 = 0.0781 w2 = 0.0000 w3 = 0.9129 Value = -8.003892e+04 
    ## 
    ##  Best Parameters Found: 
    ## Round = 22   w1 = 0.0000 w2 = 0.0000 w3 = 1.0000 Value = 16.2381

The solution has a Sharpe Ratio of 16.2381 with the following weight.

``` r
(rbayes_result <- data.frame(stock = unique(nyse$symbol),
                             weight = rbayes_finance$Best_Par) %>%
                  arrange(desc(weight)) %>%
                  mutate(weight = percent(weight, accuracy = 0.01)) %>%
                  left_join(securities, by = "stock") %>%
                  select(stock, Security, everything()))
```

    ##   stock                             Security  weight
    ## 1  ULTA Ulta Salon Cosmetics & Fragrance Inc 100.00%
    ## 2   NFX              Newfield Exploration Co   0.00%
    ## 3  ORLY                  O'Reilly Automotive   0.00%

The solution has a higher Sharpe ratio than our implementation. Both
implementations agree that ULTA is the most profitable asset.

## Particle Swarm Optimization

Let’s compare the optimum Sharpe ratio from Bayesian Optimization with
another algorithm: Particle Swarm Optimization.

We need to redefine the fitness function to suit the PSO from `pso`
package.

``` r
fitness <- function(x){
  # Assign weight for each stocks
  weight_stock <- numeric()
  for (i in 1:n_distinct(nyse$symbol)) {
    weight_stock[i] <- x[i]
  }
  
 # Calculate the numerator
 f1 <- numeric()
 for (i in 1:n_distinct(nyse$symbol)) {
   f1[i] <- weight_stock[i]*mean_stock$mean[i]
 }
   
 # Calculate the denominator
 f2 <- numeric()
 for (i in 1:n_distinct(nyse$symbol)) {
   f3 <- numeric()
   
   for (j in 1:n_distinct(nyse$symbol)) {
    f3[j] <- weight_stock[i]*weight_stock[j]*nyse_cov[i,j]
   }
   
 f2[i] <- sum(f3)
 }

  # Calculate Fitness Value
 fitness <- (sum(f1)-rf)/sum(f2)

 # Penalize Constraint Violation
 fitness <- fitness - 1e9 * (round(sum(weight_stock),10)-1)^2
 
 return(fitness)
}
```

Let’s run the PSO Algorithm. PSO will run for 10,000 iterations with
swarm size of 100. If in 500 iterations there is no improvement on the
fitness value, the algorithm will stop.

``` r
set.seed(123)
pso_finance <- psoptim(par = rep(NA,3), fn = function(x){-fitness(x)}, 
        lower = rep(0,3), upper = rep(1,3), 
        control = list(maxit = 10000, s = 100, maxit.stagnate = 500))
```

``` r
pso_finance
```

    ## $par
    ## [1] 0.18286098 0.01961205 0.79752697
    ## 
    ## $value
    ## [1] -19.2006
    ## 
    ## $counts
    ##  function iteration  restarts 
    ##    107700      1077         0 
    ## 
    ## $convergence
    ## [1] 4
    ## 
    ## $message
    ## [1] "Maximal number of iterations without improvement reached"

The solution has a Sharpe Ratio of 19.2006.

Let’s check the total weight

``` r
sum(pso_finance$par)
```

    ## [1] 1

Based on PSO, here is how your asset should be distributed.

``` r
(pso_result <- data.frame(stock = unique(nyse$symbol),
                          weight = pso_finance$par) %>% 
                arrange(desc(weight)) %>% 
                mutate(weight = percent(weight, accuracy = 0.01)) %>% 
                left_join(securities, by = "stock") %>% 
                select(stock, Security, everything()))
```

    ##   stock                             Security weight
    ## 1  ULTA Ulta Salon Cosmetics & Fragrance Inc 79.75%
    ## 2   NFX              Newfield Exploration Co 18.29%
    ## 3  ORLY                  O'Reilly Automotive  1.96%

For this problem, PSO works better than Bayesian Optimization, indicated
by the optimum fitness value. However, we only ran 30 function
evalutions (20 from samples, 10 from iterations) with Bayesian
Optimization, compared to PSO, which run more than 1000 evaluations. The
trade-off is Bayesian Optimization ran slower than PSO, since the
function evaluation is cheap.

## Normalization

We could also normalize our search grid to ensure that the weights don’t
add up to more than 1, therefore not violating the constraint.

``` r
normalized_search_grid <- normalize(search_grid)
(bayes_finance_norm <- bayesian_optimization(FUN=sharpe_ratio, lower=lower, upper=upper,
                                        init_grid_dt=normalized_search_grid, init_points=0, n_iter=1))
```

    ## $par
    ##        w1        w2        w3 
    ## 0.1336571 0.1222349 0.7441080 
    ## 
    ## $value
    ## [1] 20.13238

The solution has a Sharpe Ratio of 20.1324. We achieve a higher
performance than both `rBayesianOptimization` and Particle Swarm
Optimization, and in just one iteration\!

Based on normalized Bayes, here is how your asset should be distributed.

``` r
(bayes_norm_result <- data.frame(stock = unique(nyse$symbol),
                                 weight = bayes_finance_norm$par) %>%
                  arrange(desc(weight)) %>%
                  mutate(weight = percent(weight, accuracy = 0.01)) %>%
                  left_join(securities, by = "stock") %>%
                  select(stock, Security, everything()))
```

    ##   stock                             Security weight
    ## 1  ULTA Ulta Salon Cosmetics & Fragrance Inc 74.41%
    ## 2   NFX              Newfield Exploration Co 13.37%
    ## 3  ORLY                  O'Reilly Automotive 12.22%

``` r
write_csv(bayes_result, here("results", "bayes_result.csv"))
write_csv(rbayes_result, here("results", "rbayes_result.csv"))
write_csv(pso_result, here("results", "pso_result.csv"))
write_csv(bayes_norm_result, here("results", "bayes_norm_result.csv"))
```
