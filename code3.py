# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:07:54 2019

@author: UNR Math Stat
"""
import numpy
# from numpy import random
# from numpy import linalg
import math
from scipy import stats
import pandas
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from statsmodels import api


# This is normality testing
def normal(x):
    print('Shapiro-Wilk p =', stats.shapiro(x)[1])
    print('Jarque-Bera p =', stats.jarque_bera(x)[1])
    print('QQ plot')
    qqplot(x, line='s')
    pyplot.show()
    return 0


# This code might be useful to do regression with many factors
# It involves calculation of residuals, standard error, and normality testing
def Regression(X1, Y):
    n = numpy.size(Y)
    X = pandas.DataFrame({'1': X1})
    X = api.add_constant(X)
    Reg = api.OLS(Y.astype(float), X.astype(float)).fit()
    Y_Predictions = Reg.predict(X.astype(float))
    print(Reg.summary())
    # print([Decimal(Reg.params[k]) for k in range(3)])
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1.0 / (n - 2) * numpy.dot(residuals, residuals))
    print('Shapiro-Wilk p = ', stats.shapiro(residuals)[1])
    print('Jarque-Bera p =', stats.jarque_bera(residuals)[1])
    print('Standard Error =', stderr)
    qqplot(residuals, line = 's')
    pyplot.show()
    return (residuals)

#Three Variable Regression
def Regression3(X1, X2, X3, X4, Y):
    n = numpy.size(Y)
    X = pandas.DataFrame({'1': X1, '2': X2, '3': X3, '4': X4})
    X = api.add_constant(X)
    Reg = api.OLS(Y.astype(float), X.astype(float)).fit()
    Y_Predictions = Reg.predict(X.astype(float))
    print(Reg.summary())
    # print([Decimal(Reg.params[k]) for k in range(3)])
    residuals = Y - Y_Predictions
    stderr = math.sqrt(1.0 / (n - 5) * numpy.dot(residuals, residuals))
    print('Shapiro-Wilk p = ', stats.shapiro(residuals)[1])
    print('Jarque-Bera p =', stats.jarque_bera(residuals)[1])
    print('Standard Error =', stderr)
    qqplot(residuals, line = 's')
    pyplot.show()
    return (residuals)


# This is the correct function for simple linear regression.
# Built-in function for standard error returned standard error for
# the slope, not for the sigma in the error term.
# So we have to manually write it.
# In addition, we do normality testing for residuals.
def correctLin(x, y):
    n = numpy.size(x)
    r = stats.linregress(x, y)
    s = r.slope
    i = r.intercept
    corr = r.rvalue
    residuals = numpy.array([y[k] - x[k] * s - i for k in range(n)])
    stderr = math.sqrt((1.0 / (n - 2)) * numpy.dot(residuals, residuals))
    print('Regression Results')
    normal(residuals)
    print('slope, intercept, correl, stderr')
    return (s, i, corr, stderr)


# This function performs autoregression of order 1 on the array x:
# Regression of x[t] upon x[t-1]
def qqres(x):
    return correctLin(x[:-1], x[1:])

def ichi2(df, scale):
    shape = df/2
    return ((shape*scale)/numpy.random.gamma(shape))

# Bayesian Regression
def BayesSim(coeff, vhat, gramMatrix_inverse, df):
    simVar = ichi2(df, vhat) # Simulated variance
    simCoeff = numpy.random.multivariate_normal(coeff, simVar*gramMatrix_inverse) # Sim coefficient
    return (simCoeff)

# read the data
df1 = pandas.read_excel('ShillerMonthlyModified.xlsx')
df2 = pandas.read_excel('FRED3MonthRateMonthly.xlsx')
data1 = df1.values
data2 = df2.values
# Extracting annual (not monthly) data for columns
# Taking every 12th value
Price = data1[::12, 1]  # S&P Index price end of year (nominal)
Dividend = data1[12::12, 2]  # S&P dividend paid last year per share (nominal)
Earnings = data1[12::12, 3]  # S&P earnings per share last year (nominal)
CPI = data1[::12, 4]  # Consumer Price Index end of year
Bond = data1[::12, 5]  # 10-year Treasury rate end of year (nominal)
T = len(Dividend)  # Number of 1-year time steps
Bill = data2[:, 1]  # 3-month Treasury bill end of year (nominal)
# Total (nominal) return for S&P including reinvested dividends per year
TRStock = numpy.array([math.log(Price[k + 1] + Dividend[k]) - math.log(Price[k]) for k in range(T)])
# Total return for Treasury bill (risk-free), reinvesting every 3 months
TRBill = numpy.array([sum([math.log(1 + item / 400) for item in Bill[k * 12:12 + k * 12:3]]) for k in range(T)])
# Equity apremium
Premium = TRStock - TRBill
M = 3  # How many years in time step, to make normality correct
N = int(T / M)  # How many these steps
Years = range(1935, 2019, M)  # Every Mth year

# Cumulative nominal total return for each M-year time step for S&P
StockCum = numpy.array([sum(TRStock[M * k:M * k + M]) for k in range(N)])
# Cumulative equity premium for each multiyear time step
PremCum = numpy.array([sum(Premium[M * k:M * k + M]) for k in range(N)])
# Cumulative nominal earnings per share for S&P for each multiyear time step
EarnCum = numpy.array([sum(Earnings[M * k:M * k + M]) for k in range(N)])
# Earnings growth from last time step to this time step
EarnGrowth = numpy.array([math.log(EarnCum[k + 1] / EarnCum[k]) for k in range(N - 1)])
# Earnings growth, adjusted for inflation
RealEarnGrowth = numpy.array([EarnGrowth[k] - math.log(CPI[M * k + M] / CPI[M * k]) for k in range(N - 1)])

S = Bond[:-1]-Bill[::12] # 10-Year TB - 3-Month TB end of year, measures Market Optimism, each year starting 1934.12
Spread = S[::M] #10-Year TB - 3-Month TB for each M-Year time step

STR = Bill[::12]
ShortTermRate = STR[::M]
#print(ShortTermRate)
logSTR = numpy.log(ShortTermRate.astype('float'))
lnStrMean = numpy.mean(logSTR)
lnStrStd = numpy.std(logSTR)

t = numpy.array(range(N-1)) #Time step array
# Difference between this multiyear-step cumulative equity premium
# and real earnings growth from last multiyear period to this one
# This is expected to revert to the mean
Deviation = PremCum[1:] - RealEarnGrowth
Heat = numpy.array([sum(Deviation[:k]) for k in range(N-1)]) # Sum of Deviation up to M-year starting 1937-1938
# Test S_t regression
print('Spread Regression')
S_residual = Regression(Spread[1:-1],Spread[2:])
print('')
print('')
#Test G_t regression
print('Growth Regression')
R_residual = Regression(RealEarnGrowth[:-1],RealEarnGrowth[1:])
print('')
print('')
#Test P_t regression
print('Premium Regression')
P_residual = Regression3(Heat[:-1],t[:-1],Spread[1:-1],ShortTermRate[1:-1],PremCum[2:])
print('')
print('')
#Get Correlation Matrix of the Residuals
#print('Correlation Matrix of the Residuals')
Corr_Residual = pandas.DataFrame({'Spread': S_residual, 'REG': R_residual, 'PremCum': P_residual})
print(Corr_Residual.astype(float).corr())
#print('')
#print('Standard Error of Regressions')
#print('Spread: ', 1.204998)
#print('REG: ', 0.219148)
#print('PremCum: ', 0.220384)
#print('')

#covarianceMatrix = [[1.393939, 0.087743, 0.099498],
#					[0.087743, 0.046096, 0.023638],
#					[0.099498, 0.023638, 0.040798]]

covarianceMatrix = [[1.452020, 0.091408, 0.110800 ],
					[0.091408, 0.048026 ,0.026326 ],
					[0.110800, 0.026326, 0.048569 ]]
mean = [0,0,0]

identityMatrix =   [[1, 0, 0],
					[0, 1, 0],
					[0, 0, 1]]

print('')
print('')
#WE NEED TO ADD THE STR INTRO Simulation
#Current Value Simulation
startingSpread = Spread[-1]
startingREG = RealEarnGrowth[-1]
startingTime = t[-1]
startingHeat = Heat[-1]
startingPremCum = PremCum[-1]
simulationResults = []
for x in range(10000):
	errorTerms = numpy.random.multivariate_normal(mean, identityMatrix, 10)
	lnRandom = numpy.random.normal(lnStrMean,lnStrStd,10)
	currentSpread = startingSpread
	currentREG = startingREG
	currentTime = startingTime
	currentHeat = startingHeat
	currentPremCum = startingPremCum
	currentRun = 0
	for j in range(10): #30 Years, 3-Year time steps
		currentPremCum = 0.4371 - (0.2441*currentHeat) + (0.0236*(startingTime+j)) + (0.0130*currentSpread)-(0.0126*numpy.exp(lnRandom[j])) + errorTerms[j][2]
		currentSpread = 1.8927-(0.3743*currentSpread) + errorTerms[j][0]
		currentREG = 0.1204-(0.6043*currentREG) + errorTerms[j][1]
		currentHeat += (currentPremCum - currentREG)
		currentRun += currentPremCum
	simulationResults.append((currentRun/30.0))

print("")
ordered = numpy.sort(simulationResults)
print('Simulations with Current Values')
print('Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.mean(simulationResults))
print('')
print('STD of Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.std(simulationResults))
print('')
n, bins, patches = pyplot.hist(x=simulationResults, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
pyplot.grid(axis='y', alpha=0.75)
pyplot.xlabel('Average Equity Premium Over 3-Year Time Steps for 30 Years')
pyplot.ylabel('Frequency')
pyplot.title('Simulations with Current Values')
maxfreq = n.max()
pyplot.ylim(top=numpy.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
pyplot.show()    	
print('')

#Long Term Averages Simulation
startingSpread = numpy.mean(Spread[1:])
startingREG = numpy.mean(RealEarnGrowth[1:])
startingTime = 0
startingHeat = 0
startingPremCum = 0
simulationResults = []
for x in range(10000):
	errorTerms = numpy.random.multivariate_normal(mean, identityMatrix, 10)
	lnRandom = numpy.random.normal(lnStrMean,lnStrStd,10)
	currentSpread = startingSpread
	currentREG = startingREG
	currentTime = startingTime
	currentHeat = startingHeat
	currentPremCum = startingPremCum
	currentRun = 0

	for j in range(10): #30 Years, 3-Year time steps
		currentPremCum = 0.4371 - (0.2441*currentHeat) + (0.0236*(startingTime+j)) + (0.0130*currentSpread)-(0.0126*numpy.exp(lnRandom[j])) + errorTerms[j][2]
		currentSpread = 1.8927-(0.3743*currentSpread) + errorTerms[j][0]
		currentREG = 0.1204-(0.6043*currentREG) + errorTerms[j][1]
		currentHeat += (currentPremCum - currentREG)
		currentRun += currentPremCum
	simulationResults.append((currentRun/30))

print('Simulations with Long Term Averages')
print("")
ordered = numpy.sort(simulationResults)
print('Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.mean(simulationResults))
print('')
print('STD of Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.std(simulationResults))
print('')

n, bins, patches = pyplot.hist(x=simulationResults, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
pyplot.grid(axis='y', alpha=0.75)
pyplot.xlabel('Average Equity Premium Over 3-Year Time Steps for 30 Years')
pyplot.ylabel('Frequency')
pyplot.title('Simulations with Long Term Averages')
maxfreq = n.max()
pyplot.ylim(top=numpy.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
pyplot.show()    	
print('')

#Set up Spread info for Bayesian Inference
onesVector = numpy.ones(numpy.size(Spread[1:-1]))
spreadVector = Spread[1:-1]
gramMatrix_Spread = [[onesVector.dot(onesVector), onesVector.dot(spreadVector)],
                     [onesVector.dot(spreadVector), (spreadVector).dot(spreadVector)]]

spreadPointEstimate = [1.8927, -0.3743] # [B_0, B_1]
spreadV_hat = (1.0 / (numpy.size(spreadVector) - 2)) * numpy.dot(S_residual, S_residual)
invMatrix_Spread = numpy.linalg.inv(gramMatrix_Spread)

#Set up Real Earnings Growth info for Bayesian Inference
regVector = RealEarnGrowth[:-1]
gramMatrix_Reg = [[onesVector.dot(onesVector), onesVector.dot(regVector)],
                     [onesVector.dot(regVector), (regVector).dot(regVector)]]
invMatrix_Reg = numpy.linalg.inv(gramMatrix_Reg)
regPointEstimate = [0.1204,-0.3743] # [B_2, B_3]
regV_hat = (1.0 / (numpy.size(regVector) - 2)) * numpy.dot(R_residual, R_residual)

#Set up Cumulative Premium info for Bayesian Inference
timeVector = t[:-1]
heatVector = Heat[:-1]
#Heat, T, Spread
ShortTermRate = ShortTermRate[1:-1]
gramMatrix_CEP = [[onesVector.dot(onesVector),   heatVector.dot(onesVector),   timeVector.dot(onesVector),   spreadVector.dot(onesVector) , ShortTermRate.dot(onesVector)],
                  [onesVector.dot(heatVector),   heatVector.dot(heatVector),   timeVector.dot(heatVector),   spreadVector.dot(heatVector)  ,ShortTermRate.dot(heatVector)],
                  [onesVector.dot(timeVector),   heatVector.dot(timeVector),   timeVector.dot(timeVector),   spreadVector.dot(timeVector),  ShortTermRate.dot(timeVector)],
                  [onesVector.dot(spreadVector), heatVector.dot(spreadVector), timeVector.dot(spreadVector), spreadVector.dot(spreadVector),ShortTermRate.dot(spreadVector)],
                  [onesVector.dot(ShortTermRate), heatVector.dot(ShortTermRate), timeVector.dot(ShortTermRate), spreadVector.dot(ShortTermRate),ShortTermRate.dot(ShortTermRate)]]
invMatrix_CEP = numpy.linalg.inv(gramMatrix_CEP)
cepPointEstimate = [0.4371,-0.2441,0.0236,0.0130,-0.0126]
cepV_hat = (1.0 / (numpy.size(regVector) - 5)) * numpy.dot(P_residual, P_residual)

print("------------------Simulation with Bayesian-----------------------")
#Current Value Simulation with Baysian
startingSpread = Spread[-1]
startingREG = RealEarnGrowth[-1]
startingTime = t[-1]
startingHeat = Heat[-1]
startingPremCum = PremCum[-1]
simulationResults = []
for x in range(10000):
    errorTerms = numpy.random.multivariate_normal(mean, identityMatrix, 10)
    spreadCoeff = BayesSim(spreadPointEstimate, spreadV_hat, invMatrix_Spread, 26-2)
    regCoeff = BayesSim(regPointEstimate, regV_hat, invMatrix_Reg, 26-2)
    cepCoeff = BayesSim(cepPointEstimate, cepV_hat, invMatrix_CEP, 26-5)
    lnRandom = numpy.random.normal(lnStrMean,lnStrStd,10)
    currentSpread = startingSpread
    currentREG = startingREG
    currentTime = startingTime
    currentHeat = startingHeat
    currentPremCum = startingPremCum
    currentRun = 0
    for j in range(10): #30 Years, 3-Year time steps
        currentPremCum = cepCoeff[0] + (cepCoeff[1]*currentHeat) + (cepCoeff[2]*(startingTime+j)) + (cepCoeff[3]*currentSpread) + (cepCoeff[4]*numpy.exp(lnRandom[j])) + errorTerms[j][2]
        currentSpread = spreadCoeff[0]+(spreadCoeff[1]*currentSpread) + errorTerms[j][0]
        currentREG = regCoeff[0]+(regCoeff[1]*currentREG) + errorTerms[j][1]
        currentHeat += (currentPremCum - currentREG)
        currentRun += currentPremCum
    simulationResults.append((currentRun/30.0))
print("")
print('Simulations with Current Values and Bayesian coefficients')
print('Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.mean(simulationResults))
print('')
print('Standard Deviation of Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.std(simulationResults))
print("")
ordered = numpy.sort(simulationResults)
print("90% Value at Risk: ", ordered[1000])
print("")
print("95% Value at Risk: ", ordered[500])
n, bins, patches = pyplot.hist(x=simulationResults, bins='auto', color='#0504aa',alpha=0.7, rwidth=1)
pyplot.grid(axis='y', alpha=0.75)
pyplot.xlabel('Average Equity Premium Over 3-Year Time Steps for 30 Years (Bayesian)')
pyplot.ylabel('Frequency')
pyplot.title('Simulations with Current Values')
maxfreq = n.max()
pyplot.ylim(top=numpy.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
pyplot.show()       
print('')

#Long Term Averages Simulation with Bayesian
startingSpread = numpy.mean(Spread[1:])
startingREG = numpy.mean(RealEarnGrowth[1:])
startingTime = 0
startingHeat = 0
startingPremCum = 0
simulationResults = []
for x in range(10000):
    errorTerms = numpy.random.multivariate_normal(mean, identityMatrix, 10)
    spreadCoeff = BayesSim(spreadPointEstimate, spreadV_hat, invMatrix_Spread, 26-2)
    regCoeff = BayesSim(regPointEstimate, regV_hat, invMatrix_Reg, 26-2)
    cepCoeff = BayesSim(cepPointEstimate, cepV_hat, invMatrix_CEP, 26-5)
    lnRandom = numpy.random.normal(lnStrMean,lnStrStd,10)
    currentSpread = startingSpread
    currentREG = startingREG
    currentTime = startingTime
    currentHeat = startingHeat
    currentPremCum = startingPremCum
    currentRun = 0

    for j in range(10): #30 Years, 3-Year time steps
        currentPremCum = cepCoeff[0] + (cepCoeff[1]*currentHeat) + (cepCoeff[2]*(startingTime+j)) + (cepCoeff[3]*currentSpread) + (cepCoeff[4]*numpy.exp(lnRandom[j]))+ errorTerms[j][2]
        currentSpread = spreadCoeff[0]+(spreadCoeff[1]*currentSpread) + errorTerms[j][0]
        currentREG = regCoeff[0]+(regCoeff[1]*currentREG) + errorTerms[j][1]
        currentHeat += (currentPremCum - currentREG)
        currentRun += currentPremCum
    simulationResults.append((currentRun/30))

print('Simulations with Long Term Averages and Bayesian coefficients')
print('Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.mean(simulationResults))
print('')
print('Standard Deviation of Expected Average Equity Premium Over 3-Year Time Steps')
print(numpy.std(simulationResults))
print("")
ordered = numpy.sort(simulationResults)
print("90% Value at Risk: ", ordered[1000])
print("")
print("95% Value at Risk: ", ordered[500])

n, bins, patches = pyplot.hist(x=simulationResults, bins='auto', color='#0504aa',alpha=0.7, rwidth=1)
pyplot.grid(axis='y', alpha=0.75)
pyplot.xlabel('Average Equity Premium Over 3-Year Time Steps for 30 Years (Bayesian)')
pyplot.ylabel('Frequency')
pyplot.title('Simulations with Long Term Averages')
maxfreq = n.max()
pyplot.ylim(top=numpy.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
pyplot.show()       
