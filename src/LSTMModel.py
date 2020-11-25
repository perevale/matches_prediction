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


def test_LSTM_model(data, model, matches):
    pass


def cont_eval_LSTM(data, model, epochs=100, lr_discount=0.2 , lr=0.001, batch_size=9):
    train_function = train_LSTM_model
    test_function = test_LSTM_model


    matches = data.matches.append(data.data_val, ignore_index=True)
    # matches = matches.append(data.data_test, ignore_index=True)

    for i in range(0, matches.shape[0], batch_size):
        test_function(data, model, matches.iloc[i:i + batch_size])
        train_function(data, matches.head(i + batch_size), model, epochs,
                       lr * (1 - lr_discount) ** int(i / batch_size / 50), batch_size)
        print("T:{}, loss:{}, prediction eval:{}".format(int(i / batch_size), data.running_loss[-1],
                                                         data.running_accuracy[-1]))
    acc = float(sum(data.running_accuracy)) / matches.shape[0]
    print(acc)