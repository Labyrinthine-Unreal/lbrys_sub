# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler as mini
from sklearn.model_selection import train_test_split
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv('c_force_data.csv')
#print('COLLECTEDFEATURESINCLUDEDINTHEDATASET')
X=data.drop(['Deck1_damage'],axis=1)
y=data['Deck1_damage']
# y=labelencoder.fit_transform(y)
# mini = mini()
# X=mini.fit(X)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.12)

# mini = mini()
# X = mini.fit(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.13)
from sklearn.linear_model import LinearRegression,LogisticRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('models/c-force_model.pkl','wb'))

data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ltcusd.csv')
df0,df1 = data.shape[0], data.shape[1]
print('{} Dates '.format(df0))
# data= data.drop(['Date'], axis =1)
# data = data.drop('Adj Close',axis=1)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
X= data.drop(['close'],axis=1)
y= data['close']
mini = mini()
X = mini.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('models/lit_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/lit_model.pkl','rb'))
future_x = X
# X = X[3295:3302]
# future_x = X[-1]
# x = X[:-1]
# bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ltcusd.csv')
# date = bata['time']
# date = date.tail()
# print(date)
# bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ltcusd.csv')
# date = bata['time']
# print('PREDICTED Close')
# y = model.predict(future_x)
# print(y[-1:])


data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
df0,df1 = data.shape[0], data.shape[1]
print('{} Dates '.format(df0))
# data= data.drop(['Date'], axis =1)
# data = data.drop('Adj Close',axis=1)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
X= data.drop(['close'],axis=1)
y= data['close']

X = mini.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('models/bit_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/bit_model.pkl','rb'))
future_x = X
# X = X[3295:3302]
# future_x = X[-1]
# x = X[:-1]
bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
date = bata['time']
date = date.tail()
print(date)
bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_btcusd.csv')
date = bata['time']
print('PREDICTED Close')
y = model.predict(future_x)
print(y[-1:])


eth_data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ethusd.csv')
df0,df1 = eth_data.shape[0], eth_data.shape[1]
print('{} dates'.format(df0))
# eth_data= eth_data.drop(['Date'], axis =1)
# eth_data = eth_data.drop('Adj Close',axis=1)
eth_X= eth_data.drop(['close'],axis=1)
eth_y= eth_data['close']
eth_X = mini.fit_transform(eth_X)
eth_X_train,eth_X_test,eth_y_train,eth_y_test = train_test_split(eth_X,eth_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(eth_X_train, eth_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/eth_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/eth_model.pkl','rb'))
eth_future_x = eth_X
# eth_X = eth_X[-1:]
    # future_x = X[-1]
    # x = X[:-1]
eth_bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ethusd.csv')
eth_date = eth_bata['time']
eth_date = eth_date.tail()
eth_bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_ethusd.csv')
eth_date = eth_bata['time']
print('PREDICTED Close')
eth_y = model.predict(eth_future_x)
print('accuracy {}'.format(model.score(eth_X_test,eth_y_test)))
eth_output =eth_y[-1:]
print(eth_output)

xlm_data = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_xlmusd.csv')
df0,df1 = xlm_data.shape[0], xlm_data.shape[1]
print('{} dates'.format(df0))
# xlm_data= xlm_data.drop(['Date'], axis =1)
# xlm_data = xlm_data.drop('Adj Close',axis=1)
xlm_X= xlm_data.drop(['close'],axis=1)
xlm_y= xlm_data['close']
xlm_X = mini.fit_transform(xlm_X)
xlm_X_train,xlm_X_test,xlm_y_train,xlm_y_test = train_test_split(xlm_X,xlm_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(xlm_X_train, xlm_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/xlm_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/xlm_model.pkl','rb'))
xlm_future_x = xlm_X
# xlm_X = xlm_X[-1:]
    # future_x = X[-1]
    # x = X[:-1]
xlm_bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_xlmusd.csv')
xlm_date = xlm_bata['time']
xlm_date = xlm_date.tail()
xlm_bata = pd.read_csv('data/crypto/crypto_portfolio/1m/bitfinex_xlmusd.csv')
xlm_date = xlm_bata['time']
print('PREDICTED Close')
xlm_y = model.predict(xlm_future_x)
print('accuracy {}'.format(model.score(xlm_X_test,xlm_y_test)))
xlm_output =xlm_y[-1:]
print(xlm_output)

AAPL_data = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
df0,df1 = AAPL_data.shape[0], AAPL_data.shape[1]
print('{} dates'.format(df0))
AAPL_data = AAPL_data.fillna(28.630752329973355)
AAPL_data= AAPL_data.drop(['Date'], axis =1)
AAPL_data = AAPL_data.drop('Adj Close',axis=1)
AAPL_X= AAPL_data.drop(['Close'],axis=1)
AAPL_y= AAPL_data['Close']
AAPL_X = mini.fit_transform(AAPL_X)
AAPL_X_train,AAPL_X_test,AAPL_y_train,AAPL_y_test = train_test_split(AAPL_X,AAPL_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(AAPL_X_train, AAPL_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/AAPL_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/AAPL_model.pkl','rb'))
AAPL_future_x = AAPL_X
# AAPL_X = AAPL_X[9733:9740]
    # future_x = X[-1]
    # x = X[:-1]
AAPL_bata = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
AAPL_date = AAPL_bata['Date']
AAPL_date = AAPL_date.tail()
print(AAPL_date)
AAPL_bata = pd.read_csv('data/stocks/stocks_portfolio/AAPL.csv')
AAPL_date = AAPL_bata['Date']
print('PREDICTED Close')
AAPL_y = model.predict(AAPL_future_x)
print(AAPL_y[-1:])
AAPL_output =AAPL_y[-1:]


MSFT_data = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
df0,df1 = MSFT_data.shape[0], MSFT_data.shape[1]
print('{} dates'.format(df0))
MSFT_data= MSFT_data.drop(['Date'], axis =1)
MSFT_data = MSFT_data.drop('Adj Close',axis=1)
MSFT_X= MSFT_data.drop(['Close'],axis=1)
MSFT_y= MSFT_data['Close']
MSFT_y.mean()
MSFT_X = mini.fit_transform(MSFT_X)
MSFT_X_train,MSFT_X_test,MSFT_y_train,MSFT_y_test = train_test_split(MSFT_X,MSFT_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(MSFT_X_train, MSFT_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/MSFT_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/MSFT_model.pkl','rb'))
MSFT_future_x = MSFT_X
# MSFT_X = MSFT_X[8407:8414]
    # future_x = X[-1]
    # x = X[:-1]
MSFT_bata = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
MSFT_date = MSFT_bata['Date']
MSFT_date = MSFT_date.tail()
print(MSFT_date)
MSFT_bata = pd.read_csv('data/stocks/stocks_portfolio/MSFT.csv')
MSFT_date = MSFT_bata['Date']
print('PREDICTED Close')
MSFT_y = model.predict(MSFT_future_x)
print(MSFT_y[-1:])


MSFT_output =MSFT_y[-1:]


# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler as mini
# from sklearn.model_selection import train_test_split
roku_data = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
roku_df0,roku_df1 = roku_data.shape[0], roku_data.shape[1]
print('{} dates'.format(roku_df0))
roku_data= roku_data.drop(['Date'], axis =1)
roku_data = roku_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small roku_dataset, we will train our model with all availabe roku_data.
roku_X= roku_data.drop(['Close'],axis=1)
roku_y= roku_data['Close']
roku_X = mini.fit_transform(roku_X)
# #
roku_X_train,roku_X_test,roku_y_train,roku_y_test = train_test_split(roku_X,roku_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig roku_data
regressor.fit(roku_X_train, roku_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/roku_model.pkl','wb'))
#
# # Loading model to compare the results
roku_model = pickle.load(open('models/roku_model.pkl','rb'))
roku_future_x = roku_X
# roku_X = roku_X[:]
# future_x = X[-1]
# x = X[:-1]
roku_bata = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
roku_date = roku_bata['Date']
roku_date = roku_date.tail()
roku_bata = pd.read_csv('data/stocks/stocks_portfolio/ROKU.csv')
roku_date = roku_bata['Date']
print(roku_date.tail())
print('PREDICTED Close')
roku_y = roku_model.predict(roku_future_x)
print(roku_y[-1:])

gspc_data = pd.read_csv('data/stocks/stocks_portfolio/^GSPC.csv')
gspc_df0,gspc_df1 = gspc_data.shape[0], gspc_data.shape[1]
print('{} dates'.format(gspc_df0))
gspc_data= gspc_data.drop(['Date'], axis =1)
gspc_data = gspc_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small gspc_dataset, we will train our model with all availabe gspc_data.
gspc_X= gspc_data.drop(['Close'],axis=1)
gspc_y= gspc_data['Close']
gspc_X = mini.fit_transform(gspc_X)
# #
gspc_X_train,gspc_X_test,gspc_y_train,gspc_y_test = train_test_split(gspc_X,gspc_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig gspc_data
regressor.fit(gspc_X_train, gspc_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/^GSPC_model.pkl','wb'))
#
# # Loading model to compare the results
gspc_model = pickle.load(open('models/^GSPC_model.pkl','rb'))
gspc_future_x = gspc_X
# gspc_X = gspc_X[:]
# future_x = X[-1]
# x = X[:-1]
gspc_bata = pd.read_csv('data/stocks/stocks_portfolio/^GSPC.csv')
gspc_date = gspc_bata['Date']
gspc_date = gspc_date.tail()
gspc_bata = pd.read_csv('data/stocks/stocks_portfolio/^GSPC.csv')
gspc_date = gspc_bata['Date']
print(gspc_date.tail())
print('PREDICTED Close')
gspc_y = gspc_model.predict(gspc_future_x)
print(gspc_y[-1:])

fb_data = pd.read_csv('data/stocks/stocks_portfolio/fb.csv')
fb_df0,fb_df1 = fb_data.shape[0], fb_data.shape[1]
print('{} dates'.format(fb_df0))
fb_data= fb_data.drop(['Date'], axis =1)
fb_data = fb_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small fb_dataset, we will train our model with all availabe fb_data.
fb_X= fb_data.drop(['Close'],axis=1)
fb_y= fb_data['Close']
fb_X = mini.fit_transform(fb_X)
# #
fb_X_train,fb_X_test,fb_y_train,fb_y_test = train_test_split(fb_X,fb_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig fb_data
regressor.fit(fb_X_train, fb_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/fb_model.pkl','wb'))
#
# # Loading model to compare the results
fb_model = pickle.load(open('models/fb_model.pkl','rb'))
fb_future_x = fb_X
# fb_X = fb_X[:]
# future_x = X[-1]
# x = X[:-1]
fb_bata = pd.read_csv('data/stocks/stocks_portfolio/fb.csv')
fb_date = fb_bata['Date']
fb_date = fb_date.tail()
fb_bata = pd.read_csv('data/stocks/stocks_portfolio/fb.csv')
fb_date = fb_bata['Date']
print(fb_date.tail())
print('PREDICTED Close')
fb_y = fb_model.predict(fb_future_x)
print(fb_y[-1:])

ar_data = pd.read_csv('data/stocks/stocks_portfolio/ar.csv')
ar_df0,ar_df1 = ar_data.shape[0], ar_data.shape[1]
print('{} dates'.format(ar_df0))
ar_data= ar_data.drop(['Date'], axis =1)
ar_data = ar_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small ar_dataset, we will train our model with all availabe ar_data.
ar_X= ar_data.drop(['Close'],axis=1)
ar_y= ar_data['Close']
ar_X = mini.fit_transform(ar_X)
# #
ar_X_train,ar_X_test,ar_y_train,ar_y_test = train_test_split(ar_X,ar_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig ar_data
regressor.fit(ar_X_train, ar_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/ar_model.pkl','wb'))
#
# # Loading model to compare the results
ar_model = pickle.load(open('models/ar_model.pkl','rb'))
ar_future_x = ar_X
# ar_X = ar_X[:]
# future_x = X[-1]
# x = X[:-1]
ar_bata = pd.read_csv('data/stocks/stocks_portfolio/ar.csv')
ar_date = ar_bata['Date']
ar_date = ar_date.tail()
ar_bata = pd.read_csv('data/stocks/stocks_portfolio/ar.csv')
ar_date = ar_bata['Date']
print(ar_date.tail())
print('PREDICTED Close')
ar_y = ar_model.predict(ar_future_x)
print(ar_y[-1:]) 

chk_data = pd.read_csv('data/stocks/stocks_portfolio/chk.csv')
chk_df0,chk_df1 = chk_data.shape[0], chk_data.shape[1]
print('{} dates'.format(chk_df0))
chk_data= chk_data.drop(['Date'], axis =1)
chk_data = chk_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small chk_dataset, we will train our model with all availabe chk_data.
chk_X= chk_data.drop(['Close'],axis=1)
chk_y= chk_data['Close']
chk_X = mini.fit_transform(chk_X)
# #
chk_X_train,chk_X_test,chk_y_train,chk_y_test = train_test_split(chk_X,chk_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig chk_data
regressor.fit(chk_X_train, chk_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/chk_model.pkl','wb'))
#
# # Loading model to compchke the results
chk_model = pickle.load(open('models/chk_model.pkl','rb'))
chk_future_x = chk_X
# chk_X = chk_X[:]
# future_x = X[-1]
# x = X[:-1]
chk_bata = pd.read_csv('data/stocks/stocks_portfolio/chk.csv')
chk_date = chk_bata['Date']
chk_date = chk_date.tail()
chk_bata = pd.read_csv('data/stocks/stocks_portfolio/chk.csv')
chk_date = chk_bata['Date']
print(chk_date.tail())
print('PREDICTED Close')
chk_y = chk_model.predict(chk_future_x)
print(chk_y[-1:])

grpn_data = pd.read_csv('data/stocks/stocks_portfolio/grpn.csv')
grpn_df0,grpn_df1 = grpn_data.shape[0], grpn_data.shape[1]
print('{} dates'.format(grpn_df0))
grpn_data= grpn_data.drop(['Date'], axis =1)
grpn_data = grpn_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small grpn_dataset, we will train our model with all availabe grpn_data.
grpn_X= grpn_data.drop(['Close'],axis=1)
grpn_y= grpn_data['Close']
grpn_X = mini.fit_transform(grpn_X)
# #
grpn_X_train,grpn_X_test,grpn_y_train,grpn_y_test = train_test_split(grpn_X,grpn_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig grpn_data
regressor.fit(grpn_X_train, grpn_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/grpn_model.pkl','wb'))
#
# # Loading model to compgrpne the results
grpn_model = pickle.load(open('models/grpn_model.pkl','rb'))
grpn_future_x = grpn_X
# grpn_X = grpn_X[:]
# future_x = X[-1]
# x = X[:-1]
grpn_bata = pd.read_csv('data/stocks/stocks_portfolio/grpn.csv')
grpn_date = grpn_bata['Date']
grpn_date = grpn_date.tail()
grpn_bata = pd.read_csv('data/stocks/stocks_portfolio/grpn.csv')
grpn_date = grpn_bata['Date']
print(grpn_date.tail())
print('PREDICTED Close')
grpn_y = grpn_model.predict(grpn_future_x)
print(grpn_y[-1:])

pcg_data = pd.read_csv('data/stocks/stocks_portfolio/pcg.csv')
pcg_df0,pcg_df1 = pcg_data.shape[0], pcg_data.shape[1]
print('{} dates'.format(pcg_df0))
pcg_data= pcg_data.drop(['Date'], axis =1)
pcg_data = pcg_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small pcg_dataset, we will train our model with all availabe pcg_data.
pcg_X= pcg_data.drop(['Close'],axis=1)
pcg_y= pcg_data['Close']
pcg_X = mini.fit_transform(pcg_X)
# #
pcg_X_train,pcg_X_test,pcg_y_train,pcg_y_test = train_test_split(pcg_X,pcg_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig pcg_data
regressor.fit(pcg_X_train, pcg_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/pcg_model.pkl','wb'))
#
# # Loading model to comppcge the results
pcg_model = pickle.load(open('models/pcg_model.pkl','rb'))
pcg_future_x = pcg_X
# pcg_X = pcg_X[:]
# future_x = X[-1]
# x = X[:-1]
pcg_bata = pd.read_csv('data/stocks/stocks_portfolio/pcg.csv')
pcg_date = pcg_bata['Date']
pcg_date = pcg_date.tail()
pcg_bata = pd.read_csv('data/stocks/stocks_portfolio/pcg.csv')
pcg_date = pcg_bata['Date']
print(pcg_date.tail())
print('PREDICTED Close')
pcg_y = pcg_model.predict(pcg_future_x)
print(pcg_y[-1:])

spy_data = pd.read_csv('data/stocks/stocks_portfolio/spy.csv')
spy_df0,spy_df1 = spy_data.shape[0], spy_data.shape[1]
print('{} dates'.format(spy_df0))
spy_data= spy_data.drop(['Date'], axis =1)
spy_data = spy_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small spy_dataset, we will train our model with all availabe spy_data.
spy_X= spy_data.drop(['Close'],axis=1)
spy_y= spy_data['Close']
spy_X = mini.fit_transform(spy_X)
# #
spy_X_train,spy_X_test,spy_y_train,spy_y_test = train_test_split(spy_X,spy_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig spy_data
regressor.fit(spy_X_train, spy_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/spy_model.pkl','wb'))
#
# # Loading model to compspye the results
spy_model = pickle.load(open('models/spy_model.pkl','rb'))
spy_future_x = spy_X
# spy_X = spy_X[:]
# future_x = X[-1]
# x = X[:-1]
spy_bata = pd.read_csv('data/stocks/stocks_portfolio/spy.csv')
spy_date = spy_bata['Date']
spy_date = spy_date.tail()
spy_bata = pd.read_csv('data/stocks/stocks_portfolio/spy.csv')
spy_date = spy_bata['Date']
print(spy_date.tail())
print('PREDICTED Close')
spy_y = spy_model.predict(spy_future_x)
print(spy_y[-1:]) 

tsla_data = pd.read_csv('data/stocks/stocks_portfolio/tsla.csv')
tsla_df0,tsla_df1 = tsla_data.shape[0], tsla_data.shape[1]
print('{} dates'.format(tsla_df0))
tsla_data= tsla_data.drop(['Date'], axis =1)
tsla_data = tsla_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small tsla_dataset, we will train our model with all availabe tsla_data.
tsla_X= tsla_data.drop(['Close'],axis=1)
tsla_y= tsla_data['Close']
tsla_X = mini.fit_transform(tsla_X)
# #
tsla_X_train,tsla_X_test,tsla_y_train,tsla_y_test = train_test_split(tsla_X,tsla_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig tsla_data
regressor.fit(tsla_X_train, tsla_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/tsla_model.pkl','wb'))
#
# # Loading model to comptslae the results
tsla_model = pickle.load(open('models/tsla_model.pkl','rb'))
tsla_future_x = tsla_X
# tsla_X = tsla_X[:]
# future_x = X[-1]
# x = X[:-1]
tsla_bata = pd.read_csv('data/stocks/stocks_portfolio/tsla.csv')
tsla_date = tsla_bata['Date']
tsla_date = tsla_date.tail()
tsla_bata = pd.read_csv('data/stocks/stocks_portfolio/tsla.csv')
tsla_date = tsla_bata['Date']
print(tsla_date.tail())
print('PREDICTED Close')
tsla_y = tsla_model.predict(tsla_future_x)
print(tsla_y[-1:])

xom_data = pd.read_csv('data/stocks/stocks_portfolio/xom.csv')
xom_df0,xom_df1 = xom_data.shape[0], xom_data.shape[1]
print('{} dates'.format(xom_df0))
xom_data= xom_data.drop(['Date'], axis =1)
xom_data = xom_data.drop('Adj Close',axis=1)
# #Splitting Training and Test Set
# #Since we have a very small xom_dataset, we will train our model with all availabe xom_data.
xom_X= xom_data.drop(['Close'],axis=1)
xom_y= xom_data['Close']
xom_X = mini.fit_transform(xom_X)
# #
xom_X_train,xom_X_test,xom_y_train,xom_y_test = train_test_split(xom_X,xom_y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#
# #Fitting model with trainig xom_data
regressor.fit(xom_X_train, xom_y_train)

# Saving model to disk
pickle.dump(regressor, open('models/xom_model.pkl','wb'))
#
# # Loading model to compxome the results
xom_model = pickle.load(open('models/xom_model.pkl','rb'))
xom_future_x = xom_X
# xom_X = xom_X[:]
# future_x = X[-1]
# x = X[:-1]
xom_bata = pd.read_csv('data/stocks/stocks_portfolio/xom.csv')
xom_date = xom_bata['Date']
xom_date = xom_date.tail()
xom_bata = pd.read_csv('data/stocks/stocks_portfolio/xom.csv')
xom_date = xom_bata['Date']
print(xom_date.tail())
print('PREDICTED Close')
xom_y = xom_model.predict(xom_future_x)
print(xom_y[-1:]) 

data = pd.read_csv('data/stocks/stocks_portfolio/NIO.csv')
df0,df1 = data.shape[0], data.shape[1]
print('{} Dates '.format(df0))
data= data.drop(['Date'], axis =1)
data = data.drop('Adj Close',axis=1)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
X= data.drop(['Close'],axis=1)
y= data['Close']
# mini = MinMaxScaler()
# X = mini.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('models/nio_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('models/nio_model.pkl','rb'))
future_x = X
# X = X[3295:3302]
# future_x = X[-1:]
# x = X[:-1]
bata = pd.read_csv('data/stocks/stocks_portfolio/NIO.csv')
date = bata['Date']
date = date.tail()
print(date)
bata = pd.read_csv('data/stocks/stocks_portfolio/NIO.csv')
date = bata['Date']
print('PREDICTED Close')
y = model.predict(future_x)
print(y[-1:])