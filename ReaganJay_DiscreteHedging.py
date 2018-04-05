import numpy as np
from math import log, sqrt, e
from scipy.stats import norm
from matplotlib import pyplot as plt

#The purpose of this program is to examine the consequences of dropping the assumption of continuous hedging from the Black-Scholes Model. 

#We consider a put option with the following characteristics: S = 100, K = 100, T = 1/12, r = 0.05, q = 0, v = 0.2.

#What will happen if we rehedge once per day? Four equally-spaced times per day? 

#How does the historical volatility for these strategies compare the the implied volatility of 20%?

#We examine by running 50,000 Monte Carlo Simulations on the final profit/loss of the strategies. 

#Begin by providing Black-Scholes and Delta equations for put options:

def BS_output_put(S, K, T, r, v):
	d1 = (log(S/K) + T*(r+((v**2)/2.0)))/(v*sqrt(T))
	d2 = d1 - v*sqrt(T)
	return ((K)*(e**(-r*T))*norm.cdf(-d2) - (S)*norm.cdf(-d1))
	
def delta_output_put(S, K, T, r, v):
	d1 = (log(S/K) + T*(r+((v**2)/2.0)))/(v*sqrt(T))
	return (norm.cdf(-d1))

#------------------------------------------------------------------------------------------------------------------------------------------

#The main function:

def discrete_hedging_put(S, K, T, r, v, N, n, m): 
	#S is spot, K is strike, T is time, r is discount rate, v is volatility, N is shares per option, n is number of hedging trades, m is the number of simulations.
	put = BS_output_put(S,K,T,r,v)
	errors = np.zeros(shape=(m,1)) #Array containing final error for each trial. 
	vols = np.zeros(shape=(m,1)) #Array containing volatility for each trial. 
	for j in xrange(0, m): #Each j represents a Monte-Carlo trial.
		delta_old = 0
		delta_new = delta_output_put(S,K,T,r,v)
		price_old = 0
		price_new = S #price_new and price_old are the current and previous stock price respectively. 
		t_old = 0
		t_new = delta_new*N*S #t_new and t_old are the current and previous total hedging cost.
		shares = t_new/S #b will be the total number of shares purchased in hedging.
		log_returns = np.zeros(shape=(n,1)) #for calculating historical volatility. 
		
		for i in xrange(1,n+1): #Each i represents a rehedging trade. 
			rand = np.random.normal()
			price_old = price_new
			price_new = price_old + price_old*r*(T/n) + price_old*rand*v*sqrt(T/n)
			log_returns[i-1] = log(price_new/price_old)
			delta_old = delta_new
			if T-((i)*T/n) == 0:
				delta_new = delta_output_put(price_new,K,T/n,r, v)
			else:
				delta_new = delta_output_put(price_new,K,(T-(i)*T/n),r,v)
			t_old = t_new
			t_new = N*(delta_new - delta_old)*price_new
			shares += t_new/price_new
			t_new = t_old*(e**(r*(T/n)))+t_new
		
		std_returns = np.std(log_returns)
		vols[j] = std_returns * sqrt(252*(n/21))
		cost = t_new
		
		if price_new <= K: #put buyer chooses to sell at strike price
			option_payoff = put*(e**(r*T))*N - N*K + N*price_new #We buy 100 shares at strike price, and then sell all shares at stock price
		else: #put buyer chooses not to exercise option
			option_payoff = put*(e**(r*T))*N #We do not buy further shares. 
			
		errors[j] = ((-cost-option_payoff + shares*price_new)/100)
			
	#Next, the mean error and standard deviation of error are calculated and printed:		
	mean_error = np.mean(errors)	
	std_error = np.std(errors)

	print(mean_error)
	print(std_error)

	#Finally, the histograms for final P&L and historical volatility are created:
	bins = np.arange(-1.5, 1.5, 0.1)
	plt.hist(errors, weights = np.ones_like(errors) / len(errors), bins=bins, alpha=0.5)
	plt.title('{} Rebalancing Trades'.format(n))
	plt.xlabel('Final Profit/Loss')
	plt.ylabel('Frequency')
	plt.show()

	bins2 = np.arange(0.10, 0.30, 0.01)
	plt.hist(vols, weights = np.ones_like(vols) / len(vols), bins=bins2, alpha=0.5)
	plt.title('{} Rebalancing Trades Volatility'.format(n))
	plt.xlabel('Measured Volatility')
	plt.ylabel('Frequency')
	plt.show()

print(discrete_hedging_put(100.0, 100.0, 1/12.0, 0.05, 0.2, 100, 21, 50000))
print(discrete_hedging_put(100.0, 100.0, 1/12.0, 0.05, 0.2, 100, 84, 50000))
	
#---------------------------------------------------------------------------------------------------------------------------------

#Sample output:

#For 21 trades, sample data for final profit is a mean of -0.000386822430653 and standard deviation of 0.427481090123.

#For 84 trades, sample data for final profit is a mean of 0.000550296929175 and standard deviation of 0.217568337751.

#Main finding: making four times as many rehedging trades results in approximately 1/2 the standard deviation for final profit/loss. 
#Further, upon reviewing the historical volatility histogram, we find a very similar mean/standard-deviation trend in when comparing to implied volatility of 20%. 




