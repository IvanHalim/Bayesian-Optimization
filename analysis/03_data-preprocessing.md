Data Preprocessing
================
Ivan Timothy Halim
12/3/2020

# Data Preprocessing

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

Let’s save our data for future simulations

``` r
saveRDS(nyse, here("results", "nyse_prices.rds"))
saveRDS(securities, here("results", "nyse_securities.rds"))
saveRDS(mean_stock, here("results", "nyse_mean_returns.rds"))
saveRDS(rf, here("results", "risk_free_rate.rds"))
saveRDS(nyse_cov, here("results", "nyse_covariance.rds"))
```
