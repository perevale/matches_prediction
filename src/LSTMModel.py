import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Dense
from sklearn.preprocessing import RobustScaler



def create_LSTM_model(n_teams, lstm_units=128, optimizer='adam', loss='mean_squared_error', dropout_rate=0.2, input_length=2, batchsize=9):
  model = Sequential()
  model.add(Embedding(n_teams, batchsize, input_length=input_length))
  model.add(LSTM(units=lstm_units, recurrent_dropout=dropout_rate))
  model.add(Dropout(rate=dropout_rate))
  model.add(Dense(units=1))
  model.compile(optimizer=optimizer, loss=loss)
  return model


def train_LSTM_model(model, X, y, epochs=20, batch_size=9):
  history = model.fit(X, y,
                    epochs = epochs,
                    batch_size=batch_size,
                    shuffle=False)
  return history