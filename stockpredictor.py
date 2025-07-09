import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dropout,LSTM,Dense,GRU
data = yf.download("Titan.NS", start="2005-01-01", progress=False)
dataframe = data.copy()
dataframe.columns = [col[0] for col in dataframe.columns]
dataframe
dataframe.rename(columns={'Close': 'close'}, inplace=True)
dataframe = dataframe[['close']].copy()
dataframe
dataframe["close_lag1"] = dataframe["close"].shift(1)
dataframe["close_lag2"] = dataframe["close"].shift(2)
dataframe["close_lag3"] = dataframe["close"].shift(3)


dataframe["ma_5"] = dataframe["close"].rolling(window=5).mean()
dataframe["ma_10"] = dataframe["close"].rolling(window=10).mean()


delta = dataframe["close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
dataframe["rsi_14"] = 100 - (100 / (1 + rs))


bb_mid = dataframe["close"].rolling(window=20).mean()
bb_std = dataframe["close"].rolling(window=20).std()
dataframe["bb_mid"] = bb_mid
dataframe["bb_upper"] = bb_mid + 2 * bb_std
dataframe["bb_lower"] = bb_mid - 2 * bb_std


dataframe.dropna(inplace=True)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataframe)


X = []
y = []
window_size = 60
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)
split_index = int(len(X) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
 
# LSTM
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"LSTM - MSE: {mse:.4f}")
print(f"LSTM - R² Score: {r2:.4f}")
 
#GRU
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(units=100, return_sequences=True))
model.add(GRU(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)
y_pred_gru = model.predict(X_test)
mse_gru = mean_squared_error(y_test, y_pred_gru)
r2_gru = r2_score(y_test, y_pred_gru)
print(f"GRU - MSE: {mse_gru:.4f}")
print(f"GRU - R² Score: {r2_gru:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', linewidth=2)
    plt.plot(y_pred, label='LSTM Prediction', linestyle='--')
    plt.plot(y_pred_gru, label='GRU Prediction', linestyle=':')
    plt.title("Titan Stock Prediction: LSTM vs GRU")
    plt.xlabel("Time")
    plt.ylabel("Scaled Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 
