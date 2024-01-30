import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, LSTM
from tensorflow.python.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from IPython.display import display
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

#I decided to use LSTM for this forecast because it is a technology that genuinely interests me

#This part just reads the data and formats it into a format compatible with LSTM
df = pd.read_csv("C:\\Users\\danie.DANIELS_PC\\OneDrive\\Desktop\\orders_autumn_2020.csv", encoding="latin1")
column_index = 0
df.iloc[:, column_index] = pd.to_datetime(df.iloc[:, column_index], unit="D", origin="1900-01-01")
#Groups the data by day to get "daily_orders" and renames the TIMASTAMPS into "Date"
daily_orders = df.groupby(df.iloc[:, column_index].dt.date).size().reset_index()
daily_orders.rename(columns={df.columns[column_index]: "Date", 0: "TotalOrders"}, inplace=True)

#Sets "TotalOrders as target variable"
y = daily_orders["TotalOrders"].fillna(method="ffill")
y = y.values.reshape(-1, 1)

#Scales the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

#The input sequences have to be pretty small as the data is not large
n_lookback = 30  #Length of input sequences
n_forecast = 30  #Length of output sequences (forecast period)
X = []
Y = []
for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])
X = np.array(X)
Y = np.array(Y)

#Sets model values
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=50))
model.add(Dense(n_forecast))

#Fits or "trains" the model
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X, Y, epochs=100, batch_size=16, verbose=0)

#Generates the forecast
X_ = y[- n_lookback:]
X_ = X_.reshape(1, n_lookback, 1)
Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

#Creates a DataFrame for past and future data
df_past = daily_orders.copy()
df_future = pd.DataFrame(columns=["Date", "Forecast"])
df_future["Date"] = pd.date_range(start=df_past["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future["Forecast"] = Y_.flatten()

#Convert "Date" column to datetime before plotting
results = pd.concat([df_past, df_future])
results["Date"] = pd.to_datetime(results["Date"])

#Displays the resulting forecast
results.plot(x="Date", title="Orders", figsize=(16, 8))
plt.xlabel("Date")
plt.grid(True)
plt.show()
