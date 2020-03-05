"""
@author: Dimitar Argirov -- dimitar.argirov@nl.abnamro.com

reviewer: 
version: 0.1

SUMMARY: 

TODO:
1.

ISSUES:
1.
"""
import pandas as pd

# load all data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras import models as km
from keras import layers as kl

from matplotlib import pyplot as plt
plt.style.use('bmh')
params = {'legend.fontsize': 20,
          'figure.figsize': (20, 10),
          'axes.labelsize': 22,
          'axes.titlesize': 24,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'lines.linewidth': 4,
          'lines.markersize': 10}

path_data = r"\\solon.prd\files\P\Global\Users\C64059\UserData\Downloads\household_power_consumption" \
            r"\household_power_consumption.txt"
dataset = pd.read_csv(path_data, sep=';', header=0, low_memory=False, infer_datetime_format=True,
                   parse_dates={'datetime':[0,1]}, index_col=['datetime'])

# Correctly label missings
dataset.replace('?', np.nan, inplace=True)

# Make sure the variables are all numeric
dataset = dataset.astype('float32')

# Filling the missing values with the preceding value
dataset.fillna(method='ffill', inplace=True)

# Convert the dataframe columns to lowercase
dataset.columns = dataset.columns.str.lower()

# Put the remainder of the sub-metering as a new column
# The remainder is calculated as the excess metering of Global active power minus the sub meterings
dataset['sub_metering_4'] = (dataset["global_active_power"] * 1000 / 60) - (dataset["sub_metering_1"] +
                                                                            dataset["sub_metering_2"] +
                                                                            dataset["sub_metering_3"])

# Resample the data on a daily basis
daily_data = dataset.resample('D').sum()

# CHECK THIS FUNCTION AND OPTIMIZE IT
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def split_dataset(data):
    """
    Splits the input data into test and train set. The two sets are based on full weeks
    :param data: dataframe
        The input data to be split into train and test sets
    :return: x, y: arrays
        The train and test datasets in array form
    """
    # Standard full weeks starting on Sunday and ending on Saturday are created
    x, y = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    x = np.array(np.split(x, len(x)/7))
    y = np.array(np.split(y, len(y)/7))
    return x, y


def summarize_scores(name, score, scores):
    """
    Summarize the score(s) of the evaluated model
    :param name: string
        The name of the model
    :param score: float
        The average score for the week
    :param scores: float
        The daily scores
    :return:
    """
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def to_supervised(train, n_input, n_out=7):
    """

    :param train: np.array
        The train set
    :param n_input: int

    :param n_out:
    :return:
    """
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = km.Sequential()
    model.add(kl.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(kl.Dense(100, activation='relu'))
    model.add(kl.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# split into train and test
train, test = split_dataset(daily_data.values)
# evaluate model and get scores
n_input = 7
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
fig, ax = plt.subplots()
ax.plot(scores, marker='o', linestyle="--", label='RMSE per day in kW')
ax.set_facecolor('white')
ax.set_xlabel("Day", size=15)
ax.set_ylabel("RMSE", size=15)
ax.set_xticks(np.arange(0, 7))
ax.set_xticklabels(days, rotation=-45)
fig.legend(loc="upper left")
ax.tick_params(axis='both', labelsize=12)
plt.show()
