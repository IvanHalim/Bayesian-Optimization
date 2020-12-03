Portfolio Optimization
================
Ivan Timothy Halim
12/2/2020

# Portfolio Optimization

The problem is replicated from Zhu et al.(2011). The study employed a
PSO algorithm for portfolio selection and optimization in investment
management.

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

\[
\max f(x) = \frac{\sum\limits_{i=1}^N W_i \ast r_i - R_f}{\sum\limits_{i=1}^N \sum\limits_{j=1}^N W_i \ast W_j \ast \sigma_{ij}}
\]

**Subject To**

\[
\sum\limits_{i=1}^N W_i = 1 \\
0 \leq W_i \leq 1 \\
i = 1,2,...,N
\]

  - \(N\): Number of different assets
  - \(W_i\): Weight of each stock in the portfolio
  - \(r_i\): Return of stock i
  - \(R_f\): The test available rate of return of a risk-free security
    (i.e. the interest rate on a three month U.S. Treasury bill)
  - \(\sigma_{ij}\): Covariance between returns of assets i and j

Adjusting the portfolio weights \(w_i\), we can maximize the portfolio
Sharpe Ratio in effect balancing the trade-off between maximizing the
expected return and at the same time minimizing the risk.

## Import Data

Data is acquired from New York Stock Exchange on Kaggle (
<https://www.kaggle.com/dgawlik/nyse> ). We will only use data from
January to March of 2015 for illustration.

  - `date`: date
  - `symbol`: symbol of company stock
  - `open`: price at the open of the day
  - `close`: price at the end of the day
  - `low`: lowest price of the day
  - `high`: highest price of the day
  - `volume`: number of transaction at the day

<!-- end list -->

``` r
nyse <- read_csv(here("data", "prices.csv"))
```

    ## 
    ## -- Column specification --------------------------------------------------------
    ## cols(
    ##   date = col_datetime(format = ""),
    ##   symbol = col_character(),
    ##   open = col_double(),
    ##   close = col_double(),
    ##   low = col_double(),
    ##   high = col_double(),
    ##   volume = col_double()
    ## )

``` r
nyse <- nyse %>%
    mutate(date = ymd(date)) %>%
    filter(year(date) == 2015,
           month(date) %in% c(1:3))
head(nyse)
```

    ## # A tibble: 6 x 7
    ##   date       symbol  open close   low  high   volume
    ##   <date>     <chr>  <dbl> <dbl> <dbl> <dbl>    <dbl>
    ## 1 2015-01-02 A       41.2  40.6  40.4  41.3  1529200
    ## 2 2015-01-02 AAL     54.3  53.9  53.1  54.6 10748600
    ## 3 2015-01-02 AAP    161.  159.  157.  162.    509800
    ## 4 2015-01-02 AAPL   111.  109.  107.  111.  53204600
    ## 5 2015-01-02 ABBV    65.4  65.9  65.4  66.4  5086100
    ## 6 2015-01-02 ABC     90.6  90.5  89.8  91.3  1124600

To get clearer name of company, let’s import the Ticker Symbol and
Security.

``` r
securities <- read_csv(here("Data", "securities.csv"))
```

    ## 
    ## -- Column specification --------------------------------------------------------
    ## cols(
    ##   `Ticker symbol` = col_character(),
    ##   Security = col_character(),
    ##   `SEC filings` = col_character(),
    ##   `GICS Sector` = col_character(),
    ##   `GICS Sub Industry` = col_character(),
    ##   `Address of Headquarters` = col_character(),
    ##   `Date first added` = col_date(format = ""),
    ##   CIK = col_character()
    ## )

``` r
securities <- securities %>%
    select(`Ticker symbol`, Security) %>%
    rename(stock = `Ticker symbol`)
head(securities)
```

    ## # A tibble: 6 x 2
    ##   stock Security           
    ##   <chr> <chr>              
    ## 1 MMM   3M Company         
    ## 2 ABT   Abbott Laboratories
    ## 3 ABBV  AbbVie             
    ## 4 ACN   Accenture plc      
    ## 5 ATVI  Activision Blizzard
    ## 6 AYI   Acuity Brands Inc

Let’s say I have assets in 3 different stocks. I will randomly choose
the stocks.

``` r
set.seed(13)
selected_stock <- sample(nyse$symbol, 3)

nyse <- nyse %>%
    filter(symbol %in% selected_stock)
head(nyse)
```

    ## # A tibble: 6 x 7
    ##   date       symbol  open close   low  high  volume
    ##   <date>     <chr>  <dbl> <dbl> <dbl> <dbl>   <dbl>
    ## 1 2015-01-02 NFX     26.8  26.6  26.1  27.4 4058800
    ## 2 2015-01-02 ORLY   193.  192.  191.  195.   837800
    ## 3 2015-01-02 ULTA   128.  127.  125.  129.   410800
    ## 4 2015-01-05 NFX     25.9  24.9  24.5  26.1 4016100
    ## 5 2015-01-05 ORLY   192.  189.  189.  192.   970800
    ## 6 2015-01-05 ULTA   126.  127.  126.  128.   477400

## Calculate Returns

Let’s calculate the daily returns

``` r
nyse <- nyse %>%
    select(date, symbol, close) %>%
    group_by(symbol) %>%
    rename(price = close) %>%
    mutate(price_prev = lag(price),
           returns = (price - price_prev)/price_prev) %>%
    slice(-1) %>%
    ungroup()

head(nyse)
```

    ## # A tibble: 6 x 5
    ##   date       symbol price price_prev returns
    ##   <date>     <chr>  <dbl>      <dbl>   <dbl>
    ## 1 2015-01-05 NFX     24.9       26.6 -0.0639
    ## 2 2015-01-06 NFX     24.6       24.9 -0.0120
    ## 3 2015-01-07 NFX     23.6       24.6 -0.0402
    ## 4 2015-01-08 NFX     24.0       23.6  0.0157
    ## 5 2015-01-09 NFX     24.5       24.0  0.0217
    ## 6 2015-01-12 NFX     22.7       24.5 -0.0722

Let’s calculate the mean return of each stock

``` r
mean_stock <- nyse %>%
    group_by(symbol) %>%
    summarise(mean = mean(returns))
```

    ## `summarise()` ungrouping output (override with `.groups` argument)

``` r
head(mean_stock)
```

    ## # A tibble: 3 x 2
    ##   symbol    mean
    ##   <chr>    <dbl>
    ## 1 NFX    0.00526
    ## 2 ORLY   0.00210
    ## 3 ULTA   0.00297

The value of \(R_f\) is acquired from the latest interest rate on a
three-month U.S. Treasury bill. Since the data is from 2016, we will use
data from 2015 (Use data from March 27, 2015), which is 0.04%. The rate
is acquired from <https://ycharts.com/indicators/3_month_t_bill>.

``` r
rf <- 0.04/100
```

## Covariance Matrix Between Portofolio

Calculate the covariance matrix between portofolio. First, we need to
separate the return of each portofolio into several column by spreading
them.

``` r
nyse_wide <- nyse %>%
    pivot_wider(id_cols = c(date, symbol), names_from = symbol, values_from = returns) %>%
    select(-date)

# Create Excess Return
for (symbol in unique(nyse$symbol)) {
    nyse_wide[symbol] <- nyse_wide[symbol] - as.numeric(mean_stock[mean_stock$symbol == symbol, "mean"])
}

head(nyse_wide)
```

    ## # A tibble: 6 x 3
    ##       NFX     ORLY      ULTA
    ##     <dbl>    <dbl>     <dbl>
    ## 1 -0.0692 -0.0198  -0.000527
    ## 2 -0.0173 -0.00550 -0.00454 
    ## 3 -0.0455  0.00699  0.0256  
    ## 4  0.0104  0.0161   0.00974 
    ## 5  0.0164 -0.0302  -0.00804 
    ## 6 -0.0775 -0.0140  -0.0161

Create the covariance matrix

``` r
(nyse_cov <- cov(x = nyse_wide))
```

    ##               NFX         ORLY         ULTA
    ## NFX  1.307982e-03 5.190909e-05 2.437924e-05
    ## ORLY 5.190909e-05 2.699614e-04 8.796226e-05
    ## ULTA 2.437924e-05 8.796226e-05 1.583224e-04

## Define Fitness Function

Let’s define the fitness function. We will penalize the solution that
violate the constraint. Higher penalty will increase accuracy and force
the fitness value to get closer to the feasible area.

``` r
sharpe_ratio <- function(W, noise=0) {
    
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

### Define Parameters

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

``` r
BayesianOptimization <- function(FUN, lower, upper, init_grid_dt=NULL, init_points=1,
                                 n_iter=10, xi=0.01, noise=0, max=TRUE, acq=expected_improvement) {
  
  X_train <- init_grid_dt
  
  if (init_points > 0) {
    for (i in 1:init_points) {
      X <- runif(length(lower), lower, upper)
      rappend(X_train, t(X))
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
    Y_next <- FUN(t(X_next), noise)
    
    # Add sample to previous samples
    X_train <- rappend(X_train, X_next)
    Y_train <- rappend(Y_train, Y_next)
  }
  
  best_val <- ifelse(max, max(Y_train), min(Y_train))
  best_data <- data.frame(X_train, Y_train) %>%
    filter(Y_train == best_val)
  
  par <- unlist(best_data)[-ncol(best_data)]
  
  list("par" = par, "value" = best_val)
}
```

``` r
bayes_finance_ei <- BayesianOptimization(FUN=sharpe_ratio, lower=lower, upper=upper,
                                         init_grid_dt=search_grid, init_points=10)
```

Result of the function consists of a list with 2 components:

  - par: a vector of the best hyperparameter set found
  - value: the value of metrics achieved by the best hyperparameter set

So, what is the optimum Sharpe Ratio from Bayesian optimization?

``` r
bayes_finance_ei$value
```

    ## [1] -304.2126

The greater a portfolio’s Sharpe ratio, the better its risk-adjusted
performance. If the analysis results in a negative Sharpe ratio, it
either means the risk-free rate is greater than the portfolio’s return,
or the portfolio’s return is expected to be negative.

Let’s check the total weight of the optimum result.

``` r
sum(bayes_finance_ei$par)
```

    ## [1] 1.000566

Based on Bayesian Optimization, here is how your asset should be
distributed.

``` r
data.frame(stock = unique(nyse$symbol),
           weight = bayes_finance_ei$par) %>%
  arrange(desc(weight)) %>%
  mutate(weight = percent(weight, accuracy = 0.01)) %>%
  left_join(securities, by = "stock") %>%
  select(stock, Security, everything())
```

    ##   stock                             Security weight
    ## 1  ULTA Ulta Salon Cosmetics & Fragrance Inc 60.12%
    ## 2  ORLY                  O'Reilly Automotive 39.93%
    ## 3   NFX              Newfield Exploration Co  0.00%
