# -Decoding-Market-Trends-with-Long-Short-Term-Memory
# Remote data access for pandas
import pandas_datareader as webreader
# Mathematical functions
import math
# Fundamental package for scientific computing with Python
import numpy as np
# Additional functions for analysing and manipulating data
import pandas as pd
# Date Functions
from datetime import date, timedelta, datetime
# This function adds plotting functions for calender dates
from pandas.plotting import register_matplotlib_converters
# Important package for visualization - we use this to plot the market data
import matplotlib.pyplot as plt
# Formatting dates
import matplotlib.dates as mdates
# Packages for measuring model performance / errors
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Deep learning library, used for neural networks
from keras.models import Sequential
# Deep learning classes for recurrent and regular densely-connected layers
from keras.layers import LSTM, Dense, Dropout
# EarlyStopping during model training
from keras.callbacks import EarlyStopping
# This Scaler removes the median and scales the data according to the quantile range to normalize the price data
from sklearn.preprocessing import RobustScaler
import time
import yfinance as yf
# Setting the timeframe for the data extraction
predict_time = time.monotonic()
today = date.today()
# date_today = str(today)
date_today = today.strftime('%Y-%m-%d')
date_start = '2010-01-01'

stockname = 'Gold Jun 23'
symbol = 'GC=F'
df = yf.download(symbol,
                      start=date_start,
                      end= date_today,
                      progress=False,
)
stock_info=yf.Ticker(symbol)
print(stock_info.info)
df.head()


# Create a quick overview of the dataset
train_dfs = df.copy()
train_dfs
# List of considered Features
FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume', #'Month',
            #'Adj Close'
           ]
print('FEATURE LIST')
print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_dfs)
data_filtered = data[FEATURES]

# Print the tail of the dataframe
data_filtered.tail()
# # Plot each column
# register_matplotlib_converters()
# nrows = 3
# ncols = int(round(train_dfs.shape[1] / nrows, 0))
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(16, 7))
# fig.subplots_adjust(hspace=0.3, wspace=0.3)
# x = train_dfs.index
# columns = train_dfs.columns
# f = 0
# for i in range(nrows):
#     for j in range(ncols):
#         ax[i, j].xaxis.set_major_locator(mdates.YearLocator())
#         assetname = columns[f]
#         y = train_dfs[assetname]
#         f += 1
#         ax[i, j].plot(x, y, color='#039dfc', label=stockname, linewidth=1.0)
#         ax[i, j].set_title(assetname)
#         ax[i, j].tick_params(axis="x", rotation=90, labelsize=10, length=0)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

register_matplotlib_converters()
nrows = 3
ncols = int(np.ceil(train_dfs.shape[1] / nrows))  # Use np.ceil to avoid rounding issues
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(16, 7))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x = train_dfs.index
columns = train_dfs.columns
f = 0

for i in range(nrows):
    for j in range(ncols):
        if f >= len(columns):  # Ensure f does not exceed the number of columns
            ax[i, j].axis("off")  # Hide empty subplots
            continue

        ax[i, j].xaxis.set_major_locator(mdates.YearLocator())
        assetname = columns[f]
        y = train_dfs[assetname]
        f += 1
        ax[i, j].plot(x, y, color='#039dfc', label=assetname, linewidth=1.0)  # Fix: Use assetname instead of stockname
        ax[i, j].set_title(assetname)
        ax[i, j].tick_params(axis="x", rotation=90, labelsize=10, length=0)

plt.show()
# Indexing Batches
train_df = train_dfs.sort_values(by=['Date']).copy()

# We safe a copy of the dates index, before we need to reset it to numbers
date_index = train_df.index

# Adding Month and Year in separate columns
d = pd.to_datetime(train_df.index)
train_df['Month'] = d.strftime("%m")
train_df['Year'] = d.strftime("%Y")

# We reset the index, so we can convert the date-index to a number-index
train_df = train_df.reset_index(drop=True).copy()
train_df.head(5)
#List of considered Features
FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume', 'Month',
            #'Adj Close'
           ]
print('FEATURE LIST')
print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]

# Print the tail of the dataframe
data_filtered.tail()
# Calculate the number of rows in the data
nrows = data_filtered.shape[0]
np_data_unscaled = np.array(data_filtered)
np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data_unscaled.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = RobustScaler()
np_data = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = RobustScaler()
df_Close = pd.DataFrame(data_filtered['Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)
#Settings
sequence_length = 100

# Split the training data into x_train and y_train data sets
# Get the number of rows to train the model on 80% of the data
train_data_len = math.ceil(np_data.shape[0] * 0.8) #2616

# Create the training data
train_data = np_data[0:train_data_len, :]
x_train, y_train = [], []
# The RNN needs data with the format of [samples, time steps, features].
# Here, we create N samples, 100 time steps per sample, and 2 features
for i in range(100, train_data_len):
    x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_train.append(train_data[i, 0]) #contains the prediction values for validation

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the test data
test_data = np_data[train_data_len - sequence_length:, :]

# Split the test data into x_test and y_test
x_test, y_test = [], []
test_data_len = test_data.shape[0]
for i in range(sequence_length, test_data_len):
    x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_test.append(test_data[i, 0]) #contains the prediction values for validation
    # Convert the x_train and y_train to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Convert the x_train and y_train to numpy arrays
x_test = np.array(x_test); y_test = np.array(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# Configure the neural network model
model = Sequential()

# Model with 100 Neurons
# inputshape = 100 Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=False,
               input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Training the model
epochs = 5
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=16,
                    epochs=epochs, callbacks=[early_stop])
# Plot training & validation loss values
fig, ax = plt.subplots(figsize=(5, 5), sharex=True)
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
plt.legend(["Train", "Test"], loc="upper left")
plt.grid()
plt.show()
# Get the predicted values
predictions = model.predict(x_test)

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test, predictions)/ y_test))) * 100
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test, predictions)/ y_test)) ) * 100
print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE, 2)) + ' %')


# Ensure pred_unscaled is a 1D array
pred_unscaled = scaler_pred.inverse_transform(predictions.reshape(-1, 1)).squeeze()

# The date from which the data is displayed
display_start_date = pd.Timestamp('today') - timedelta(days=500)

# Add the date column
data_filtered_sub = data_filtered.copy()
data_filtered_sub['Date'] = date_index

# Add the difference between the valid and predicted prices
train = data_filtered_sub.iloc[:train_data_len + 1].copy()
valid = data_filtered_sub.iloc[train_data_len:].copy()

# Ensure `valid` has the same length as `pred_unscaled`
valid = valid.iloc[:len(pred_unscaled)].copy()

# Insert "Prediction" as a new column
valid.insert(1, "Prediction", pred_unscaled, True)

# Compute the difference manually and insert it
valid["Difference"] = valid["Prediction"].squeeze() - valid["Close"].squeeze()


# Zoom in to a closer timeframe
valid = valid[valid['Date'] > display_start_date]
train = train[train['Date'] > display_start_date]

# Visualize the data
fig, ax1 = plt.subplots(figsize=(22, 10), sharex=True)
xt = train['Date']
yt = train[["Close"]]
xv = valid['Date']
yv = valid[["Close", "Prediction"]]

plt.title("Predictions vs Actual Values", fontsize=20)
plt.ylabel(stockname, fontsize=18)
plt.plot(xt, yt, color="#039dfc", linewidth=2.0)
plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0)
plt.plot(xv, yv["Close"], color="black", linewidth=2.0)
plt.legend(["Train", "Test Predictions", "Actual Values"], loc="upper left")

# Create the bar plot with the differences
x = valid['Date']
y = valid["Difference"]

# Create custom color range for positive and negative differences
valid.loc[y >= 0, 'diff_color'] = "#2BC97A"
valid.loc[y < 0, 'diff_color'] = "#C92B2B"

plt.bar(x, y, width=0.8, color=valid['diff_color'])
plt.grid()
plt.show()
import datetime as dt
start_time = time.monotonic()
# # Get fresh data until today and create a new dataframe with only the price data
# date_start = pd.Timestamp('today') - timedelta(days=200)
# new_df = yf.download(symbol,
#                       start=date_start,
#                       end=date_today,
#                       progress=False,
# )
# # df.head()
# # new_df = webreader.DataReader(symbol, data_source='yahoo', start=date_start, end=date_today)
# d = pd.to_datetime(new_df.index)
# new_df['Month'] = d.strftime("%m")
# new_df['Year'] = d.strftime("%Y")
# new_df = new_df.filter(FEATURES)

# # Get the last 100 day closing price values and scale the data to be values between 0 and 1
# last_100_days = new_df[-100:].values
# last_100_days_scaled = scaler.transform(last_100_days)

# # Create an empty list and Append past 100 days
# X_test_new = []
# X_test_new.append(last_100_days_scaled)

# # Convert the X_test data set to a numpy array and reshape the data
# pred_price_scaled = model.predict(np.array(X_test_new))
# pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled)
# # Print last price and predicted price for the next day
# price_today = round(new_df['Close'][-1], 2)
# predicted_price = round(pred_price_unscaled.ravel()[0], 2)
# percent = round(100 - (predicted_price * 100)/price_today, 2)

# if percent > 0:
#     a = '-'
# else:
#     a = '+'

# print('The close price for ' + stockname + ' at ' + str(today) + ' was: ' + str(price_today))
# print('The predicted close price is: ' + str(pred_price_unscaled) + ' (' + a + str(percent) + '%)')
# end_time = time.monotonic()
# predict_end_time = time.monotonic()
# print(dt.timedelta(seconds=end_time - start_time))
# print(dt.timedelta(seconds=predict_end_time-predict_time))
# Select 'Close' column correctly from MultiIndex DataFrame
price_today = round(new_df[('Close', symbol)].iloc[-1], 2)

predicted_price = round(pred_price_unscaled.ravel()[0], 2)

# Ensure percent is a scalar, not a Series
percent = round(100 - (predicted_price * 100) / price_today, 2)

# Fix comparison by explicitly converting percent to a number
a = '-' if percent > 0 else '+'

print(f'The close price for {stockname} at {today} was: {price_today}')
print(f'The predicted close price is: {predicted_price} ({a}{abs(percent)}%)')
def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the LSTM model based on Mean Absolute Percentage Error (MAPE).

    Parameters:
    y_true (numpy array): Array of true values.
    y_pred (numpy array): Array of predicted values.

    Returns:
    accuracy (float): Accuracy of the model.
    """
    MAPE = np.mean((np.abs(np.subtract(y_true, y_pred) / y_true))) * 100
    accuracy = 100 - MAPE
    return accuracy

# Calculate accuracy
accuracy = calculate_accuracy(y_test, predictions)
print('Accuracy of the LSTM model:', round(accuracy, 2), '%')
# Configure the neural network model
model = Sequential()

# Model with 100 Neurons
# inputshape = 100 Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=False,
               input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1, activation='relu'))

# Compile the model with RMSprop optimizer
model.compile(optimizer='rmsprop', loss='mean_squared_error')
# Training the model
epochs = 5
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=16,
                    epochs=epochs, callbacks=[early_stop])
# Plot training & validation loss values
fig, ax = plt.subplots(figsize=(5, 5), sharex=True)
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
plt.legend(["Train", "Test"], loc="upper left")
plt.grid()
plt.show()
# Get the predicted values
predictions = model.predict(x_test)

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test, predictions)/ y_test))) * 100
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test, predictions)/ y_test)) ) * 100
print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE, 2)) + ' %')
# # Get the predicted values
# pred_unscaled = scaler_pred.inverse_transform(predictions)

# # The date from which on the date is displayed
# display_start_date = pd.Timestamp('today') - timedelta(days=500)

# # Add the date column
# data_filtered_sub = data_filtered.copy()
# data_filtered_sub['Date'] = date_index

# # Add the difference between the valid and predicted prices
# train = data_filtered_sub[:train_data_len + 1]
# valid = data_filtered_sub[train_data_len:]
# valid.insert(1, "Prediction", pred_unscaled.ravel(), True)
# valid.insert(1, "Difference", valid["Prediction"] - valid["Close"], True)

# # Zoom in to a closer timeframe

# valid = valid[valid['Date'] > display_start_date]
# train = train[train['Date'] > display_start_date]

# # Visualize the data
# fig, ax1 = plt.subplots(figsize=(22, 10), sharex=True)
# xt = train['Date']; yt = train[["Close"]]
# xv = valid['Date']; yv = valid[["Close", "Prediction"]]
# plt.title("Predictions vs Actual Values", fontsize=20)
# plt.ylabel(stockname, fontsize=18)
# plt.plot(xt, yt, color="#039dfc", linewidth=2.0)
# plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0)
# plt.plot(xv, yv["Close"], color="black", linewidth=2.0)
# plt.legend(["Train", "Test Predictions", "Actual Values"], loc="upper left")

# # # Create the bar plot with the differences
# x = valid['Date']
# y = valid["Difference"]

# # Create custom color range for positive and negative differences
# valid.loc[y >= 0, 'diff_color'] = "#2BC97A"
# valid.loc[y < 0, 'diff_color'] = "#C92B2B"

# plt.bar(x, y, width=0.8, color=valid['diff_color'])
# plt.grid()
# plt.show()
# Ensure column names are strings
valid.columns = valid.columns.astype(str)

# Identify Prediction, Close, and Date columns
prediction_cols = [col for col in valid.columns if col == "P"]
close_cols = [col for col in valid.columns if col == "C"]
date_cols = [col for col in valid.columns if col == "D"]

# Debugging: Print identified columns
print("Identified Prediction column:", prediction_cols[0] if prediction_cols else "None")
print("Identified Close column:", close_cols[0] if close_cols else "None")
print("Identified Date column:", date_cols[0] if date_cols else "None")

# Select the first 'Prediction' column safely (extract only first column)
if prediction_cols:
    valid["Prediction"] = valid[prediction_cols].iloc[:, 0]  # Extract first column

# Select the first 'Close' column safely (extract only first column)
if close_cols:
    valid["Close"] = valid[close_cols].iloc[:, 0]  # Extract first column

# Select the first 'Date' column and convert it to datetime
if date_cols:
    valid["Date"] = pd.to_datetime(valid[date_cols].iloc[:, 0], errors="coerce")

# Ensure 'Close' and 'Prediction' are numeric
valid["Close"] = pd.to_numeric(valid["Close"], errors="coerce")
valid["Prediction"] = pd.to_numeric(valid["Prediction"], errors="coerce")

# Compute the Difference
valid["Difference"] = valid["Prediction"] - valid["Close"]

# Zoom into the last 500 days
display_start_date = pd.Timestamp('today') - timedelta(days=500)
valid = valid[valid["Date"] > display_start_date]
train = train[train["Date"] > display_start_date]

# Visualization
fig, ax1 = plt.subplots(figsize=(22, 10), sharex=True)
xt = train["Date"]
yt = train["Close"]
xv = valid["Date"]
yv = valid[["Close", "Prediction"]]

plt.title("Predictions vs Actual Values", fontsize=20)
plt.ylabel("Stock Price", fontsize=18)
plt.plot(xt, yt, color="#039dfc", linewidth=2.0, label="Train Data")
plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0, label="Predictions")
plt.plot(xv, yv["Close"], color="black", linewidth=2.0, label="Actual Values")
plt.legend(loc="upper left")

# Bar plot for differences
x = valid["Date"]
y = valid["Difference"]

# Define colors for positive/negative differences
valid["diff_color"] = valid["Difference"].apply(lambda val: "#2BC97A" if val >= 0 else "#C92B2B")

plt.bar(x, y, width=0.8, color=valid["diff_color"])
plt.grid()
plt.show()
import datetime as dt
start_time = time.monotonic()
import os
print("Files in current directory:", os.listdir())
# Get fresh data until today and create a new dataframe with only the price data
date_start = pd.Timestamp('today') - timedelta(days=200)
new_df = yf.download(symbol,
                      start=date_start,
                      end=date_today,
                      progress=False,
)
# df.head()
# new_df = webreader.DataReader(symbol, data_source='yahoo', start=date_start, end=date_today)
d = pd.to_datetime(new_df.index)
new_df['Month'] = d.strftime("%m")
new_df['Year'] = d.strftime("%Y")
new_df = new_df.filter(FEATURES)

# Get the last 100 day closing price values and scale the data to be values between 0 and 1
last_100_days = new_df[-100:].values
last_100_days_scaled = scaler.transform(last_100_days)

# Create an empty list and Append past 100 days
X_test_new = []
X_test_new.append(last_100_days_scaled)

# Convert the X_test data set to a numpy array and reshape the data
pred_price_scaled = model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled)
# Print last price and predicted price for the next day
price_today = round(new_df['Close'][-1], 2)
predicted_price = round(pred_price_unscaled.ravel()[0], 2)
percent = round(100 - (predicted_price * 100)/price_today, 2)

if percent > 0:
    a = '-'
else:
    a = '+'

print('The close price for ' + stockname + ' at ' + str(today) + ' was: ' + str(price_today))
print('The predicted close price is: ' + str(pred_price_unscaled) + ' (' + a + str(percent) + '%)')
end_time = time.monotonic()
predict_end_time = time.monotonic()
print(dt.timedelta(seconds=end_time - start_time))
print(dt.timedelta(seconds=predict_end_time-predict_time))

def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the LSTM model based on Mean Absolute Percentage Error (MAPE).

    Parameters:
    y_true (numpy array): Array of true values.
    y_pred (numpy array): Array of predicted values.

    Returns:
    accuracy (float): Accuracy of the model.
    """
    MAPE = np.mean((np.abs(np.subtract(y_true, y_pred) / y_true))) * 100
    accuracy = 100 - MAPE
    return accuracy

# Calculate accuracy
accuracy = calculate_accuracy(y_test, predictions)
print('Accuracy of the LSTM model:', round(accuracy, 2), '%')
