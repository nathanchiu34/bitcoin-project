#################################################################

# BITCOIN PROJECT - DEC. 28th 2015

# Nathan Chiu

# Personal Project for Bitcoin Analysis. Feel free to contribute to it. 

##################################################################

import Quandl
import numpy as np 
import pandas as pd 
import operator
import statsmodels.formula.api as sm 
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
#from matplotlib import pyplot as plt

# We know that in 2013 Bitcoin grew in price, then in 2014 it crashed.
# Let's examine the given data to see what happened.

##############################################################################
# Download the data from Quandl
#df_bitcoin = Quandl.get("BCHARTS/MTGOXUSD",authtoken="P5_mu65sQdbNKTppzVE8")
df_bitcoin = Quandl.get("BCHARTS/LOCALBTCUSD")
df_sp500 = Quandl.get("YAHOO/INDEX_GSPC",authtoken="P5_mu65sQdbNKTppzVE8")

###############################################################################
#print df_bitcoin.head()
#print df_bitcoin.corr()
# This line of code prints the bitcoin data as a numpy array
#print df_bitcoin1[0:10]

# Split up the Quandl data into 2013 data
df_bitcoin13 = pd.DataFrame() 
for i in df_bitcoin.index:
	if i.year == 2013: 
		#print 'true'
		#print type(df_bitcoin.loc[str(i)])
		df_bitcoin13 = df_bitcoin13.append(df_bitcoin.loc[str(i)])#ignore_index=True)
#bitcoin 2013 data as a numpy array
df_bitcoin131 = np.array(df_bitcoin13)

# Split up the Quandl data into 2014 data
df_bitcoin14 = pd.DataFrame()
for i in df_bitcoin.index:
	if i.year == 2014:
		df_bitcoin14 = df_bitcoin14.append(df_bitcoin.loc[str(i)])
#bitcoin 2104 data as a numpy array
df_bitcoin141 = np.array(df_bitcoin14)

# Split up Quandl SP 500 data into 2013 dat
df_sp500_13 = pd.DataFrame()
for i in df_sp500.index:
	if i.year == 2013:
		df_sp500_13 = df_sp500_13.append(df_sp500.loc[str(i)])

'''
##########################################################################

# Function below automates above code process of splitting into years

###########################################################################
years = {'2012': 2012, '2013': 2013, '2014': 2014, '2015': 2015}
df_bitcoin12, df_bitcoin13, df_bitcoin14, df_bitcoin15 = pd.DataFrame()
df_sp500_12, df_sp500_13, df_sp500_14, df_sp500_15 = pd.DataFrane()

def data_year_split():
	for i in years:
		pd.DataFrame()

'''
###########################################################################

#GET DESCRIPTIVE STATISTICS

###########################################################################
def get_stats():
	print "BITCOIN 2013 STATISTICS****************"
	print df_bitcoin13.describe()
	print " "
	print "BITCOIN 2014 STATISTICS****************"
	print df_bitcoin14.describe()

def get_corr():
	print "BITCOIN 13 SP13 CORRELATIONS"
	print np.correlate(df_bitcoin13['Volume (BTC)'],df_sp500_13['Volume'])
	print " "
	print "BITCOIN 14 SP14 CORRELATIONS"
	print np.correlate(df_bitcoin14['Volume (BTC)'],df_sp500_14['Volume'])
############################################################################ 

# DATA VISUALIZATION
# Printing of arrays and diagrams

#########################################################################
def show_2014():
	#Volume 13 and 14
	vol_SP500 = np.array(df_sp500_13['Volume'])
	vol_Bitcoin13 = np.array(df_bitcoin13['Volume (BTC)'])
	vol_Bitcoin14 = np.array(df_bitcoin14['Volume (BTC)'])
	vol_Bitcoin_index = np.array(df_bitcoin13.index)
	arraySP500_index = np.arange(vol_SP500.size)
	arrayBitcoin_index13 = np.arange(vol_Bitcoin13.size)
	arrayBitcoin_index14 = np.arange(vol_Bitcoin14.size)
	#plt.scatter( arraySP500_index, vol_SP500, marker = 'o', color = 'green')
	plt.scatter( arrayBitcoin_index13, vol_Bitcoin13, marker = 'x', color= 'blue')
	plt.scatter( arrayBitcoin_index14, vol_Bitcoin14, marker = 'o', color ='green')
	plt.show()

def show_adjustedVolume():
	# Adjusted Volume / Mean Volume for S&P 500 and Bitcoin
	# plt.scatter(df_sp500_13['Volume'],df_bitcoin13['Volume'])
	adj_vol_SP500 = np.array(df_sp500_13['Volume'] / df_sp500_13['Volume'].mean())
	adj_vol_Bitcoin = np.array(df_bitcoin13['Volume (BTC)']/ df_bitcoin13['Volume (BTC)'].mean())
	arraySP500_index = np.arange(adj_vol_SP500.size)
	arrayBitcoin_index = np.arange(adj_vol_Bitcoin.size)

	plt.scatter(arraySP500_index,adj_vol_SP500,marker = 'o',color='green') 
	plt.scatter(arrayBitcoin_index,adj_vol_Bitcoin,marker = 'x',color='blue')
	#plt.scatter(df_bitcoin13.index,array1)
	plt.title('Volume / Mean Volume for S&P 500 vs. Bitcoin')
	plt.ylabel('Bitcoin')
	plt.xlabel('S&P 500')
	plt.show()

#################################################################################

#### BUILDING A LINEAR REGRESSION MODEL FOR 2013 DATA

#### MULTI VARIATE GIVEN OPEN, CLOSE, VOLUME(BTC), TO ESTIMATE WEIGHTED PRICE

#### DEPSITE 2014'S CRASH, PERHAP IF GROWTH CONTINUES INTO 2015, MODELS CAN THEN BE
#### REASSESED FOR VALIDITIY
#################################################################################

#Split up the 2013 data into these different arrays
open_13 = np.array(df_bitcoin13['Open'])
close_13 = np.array(df_bitcoin13['Close'])
vol_13 = np.array(df_bitcoin13['Volume (BTC)'])
price_13 = np.array(df_bitcoin13['Weighted Price'])

open_14 = np.array(df_bitcoin14['Open'])
close_14 = np.array(df_bitcoin14['Close'])
vol_14 = np.array(df_bitcoin14['Volume (BTC)'])
price_14 = np.array(df_bitcoin14['Weighted Price'])

print len(df_bitcoin14)
print len(df_bitcoin13)
#print close_13
#print vol_13
#print price_13
#clf = linear_model.LinearRegression()
#clf.fit((open_13, close_13, vol_13), price_13)
#clf.fit(open_13, price_13)
