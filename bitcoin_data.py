import Quandl
import numpy as np 
import pandas as pd 
import operator
import statsmodels.formula.api as sm 
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 

#THE PLAN:
# 1. CREATE A PREDICTION MODEL FOR 2013 BITCOIN DATA
# 2. SEE HOW IT MATCHES UP TO THE 2014 BITCOIN DATA
# 3. CHOOSE THE BEST LOOKING MODEL FOR 2014 DATA
# 4. USE THE 2014 DATA TO ESTIMATE 2015 POSSIBLE ANALYSIS

# Download the data from Quandl
df_bitcoin = Quandl.get("BCHARTS/ICBITUSD",authtoken="P5_mus65sQdbNKTppzVE8")
df_bitcoin1 = np.array(df_bitcoin)
#print df_bitcoin.head()
#print df_bitcoin.corr()
# This line of code prints the bitcoin data as a numpy array
#print df_bitcoin1[0:10]

# Split up the Quandl data into 2013 data
df_bitcoin13 = pd.DataFrame() 
for i in df_bitcoin.index:
	#print i.year
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

#some printing of arrays and diagrams
#plt.scatter(df_bitcoin13.index,df_bitcoin13['Weighted Price'])
#plt.scatter(df_bitcoin13.index,array1)
#plt.show()

#################################################################################

#### BUILDING A LINEAR REGRESSION MODEL FOR 2013 DATA

#    MULTI VARIATE GIVEN OPEN, CLOSE, VOLUME(BTC), TO ESTIMATE WEIGHTED PRICE
#################################################################################

#Split up the 2013 data into these different arrays
open_13 = np.array(df_bitcoin13['Open'])
close_13 = np.array(df_bitcoin13['Close'])
vol_13 = np.array(df_bitcoin13['Volume (BTC)'])
price_13 = np.array(df_bitcoin13['Weighted Price'])

clf = linear_model.LinearRegression()
clf.fit((open_13, close_13, vol_13), price_13)

