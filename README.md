# LongTerm
This is a repository of our updated code and data for an article on long-term modeling of the stock market.

Code2 is a regression model of Spread (between 10 year and 3 month Treasury rates), Equity Premium (total return of S&P Composite index minus risk-free return of 3 month Treasuries), and Real Earnings Growth (adjusted for inflation) in 3 year steps over 1935-2018. We model Equity Premium using Heat: an additional variable computing how much the market is overpriced, based on comparing past Equity Premia vs past Real Earnings Growth. This code also contains simulations with initial market conditions (Real Earnings Growth, Spread, and Heat) set as current (as of end of 2018) and long-term historical averages. Simulations are both standard and Bayesian, with non-informative Jeffrey's prior. 

Code3 contains the same but with short-term (3 month) interest rate modeled using reasonable guesses. Regression of Equity Premium here also includes this interest rate as factor. We cannot model this rate as autoregression because it is highly correlated with Fed funds rate, which is decided by the Fed, not so much by the market.
