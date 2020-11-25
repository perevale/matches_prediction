import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Dense
from sklearn.preprocessing import RobustScaler
from keras.optimizers import Adam
import tensorflow as tf


def create_LSTM_model(n_teams, lstm_units=128, optimizer='adam', loss='sparse_categorical_crossentropy', dropout_rate=0.2, input_length=2, batchsize=9):
    model = Sequential()
    model.add(Embedding(n_teams, batchsize, input_length=input_length))
    model.add(LSTM(units=lstm_units, recurrent_dropout=dropout_rate))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=3))
    model.compile(optimizer=Adam(lr=0.01), loss=loss, metrics=['categorical_accuracy'])
    return model


def train_LSTM_model(data, model, X, y, epochs=10, batch_size=9 ):
    history = model.fit(X, y,
                    epochs = epochs,
                    batch_size=batch_size,
                    shuffle=False)
    return history


def test_LSTM_model(data, model, X, y):
    loss, accuracy = model.evaluate(X, y, batch_size=9)
    data.running_loss.append(loss)
    data.running_accuracy.append(accuracy)
    return accuracy


def cont_eval_LSTM(data, model, epochs=10, lr_discount=0.2 , lr=0.001, batch_size=9):
    tf.keras.backend.clear_session()

    train_function = train_LSTM_model
    test_function = test_LSTM_model


    matches = data.matches.append(data.data_val, ignore_index=True)
    # matches = matches.append(data.data_test, ignore_index=True)

    for i in range(0, matches.shape[0], batch_size):
        X_test, y_test = matches.iloc[i:i + batch_size][['home_team', 'away_team']].to_numpy().astype('float32'),\
                         matches.iloc[i:i + batch_size][['lwd']].to_numpy().astype('int64')
        X, y = matches.head(i + batch_size)[['home_team', 'away_team']].to_numpy().astype('float32'),\
               matches.head(i + batch_size)[['lwd']].to_numpy().astype('int64')
        test_function(data, model, X_test, y_test)
        train_function(data, model, X, y, epochs, batch_size)
        print("T:{}, loss:{}, acc:{}".format(int(i / batch_size), data.running_loss[-1],
                                                         data.running_accuracy[-1]))
    acc = float(sum(data.running_accuracy)) / len(data.running_accuracy)
    print(acc)

    X, y = data.data_test[['home_team', 'away_team']].to_numpy().astype('float32'),\
        data.data_test[['lwd']].to_numpy().astype('int64')
    data.test_accuracy = test_function(data, model, X, y)
    print("Test accuracy:{}".format(data.test_accuracy))