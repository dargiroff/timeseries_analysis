from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
from sklearn.metrics import mean_squared_error

from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import ScalarFormatter

register_matplotlib_converters()

plt.style.use('bmh')
plt.rcParams.update({"axes.facecolor": "white"})

url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
      r'csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
df = pd.read_csv(url)
df_non_china = df[df['Country/Region'] != 'China']
df = df.iloc[:, 4:]
df_non_china = df_non_china.iloc[:, 4:]

com_daily_cases = df.sum(axis=0)
com_daily_cases.index = pd.to_datetime(com_daily_cases.index)
daily_cases = com_daily_cases.diff().fillna(com_daily_cases[0]).astype(np.int64)


def split_train_test(data, pct_test):
    test_data_size = int(pct_test * len(data))
    train_data = data[:-test_data_size]
    test_data = data[-test_data_size:]
    return train_data, test_data


def build_model(train_data, test_data, **kwargs):
    model = ExponentialSmoothing(train_data, trend=kwargs['trend'], damped=kwargs['damped'],
                                 seasonal=kwargs['seasonal'], seasonal_periods=kwargs['seasonal_periods'])
    model_fit = model.fit(optimized=kwargs['optimized'], use_boxcox=kwargs['use_boxcon'],
                          remove_bias=kwargs['remove_bias'])
    pred = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
    return pred


def score_model(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def select_params(train_data, test_data, grid):
    t_params = grid['trend']
    d_params = grid['damped']
    s_params = grid['seasonal']
    p_params = grid['seasonal_periods']
    o_params = grid['optimized']
    b_params = grid['use_boxcon']
    r_params = grid['remove_bias']

    iter_product = itertools.product(t_params, d_params, s_params, p_params, o_params, b_params, r_params)
    combos = list(map(list, iter_product))

    scores = pd.DataFrame(columns=['score', 'paramlist'])
    for idx, paramlist in enumerate(combos):
        try:
            prediction = build_model(train_data, test_data, trend=paramlist[0], damped=paramlist[1],
                                     seasonal=paramlist[2],
                                     seasonal_periods=paramlist[3], optimized=paramlist[4], use_boxcon=paramlist[5],
                                     remove_bias=paramlist[6])
            score = score_model(test_data, prediction)
            scores.loc[idx, 'score'] = score
            scores.loc[idx, 'paramlist'] = paramlist
        except (TypeError, ValueError):
            print('Unable to build model with specs: ', paramlist)
            pass

    scores = scores.sort_values('score').reset_index()

    return scores


train, test = split_train_test(daily_cases, 0.33)

grid = dict()
grid['trend'] = ['add', 'mul', None]
grid['damped'] = [True, False]
grid['seasonal'] = ['add', 'mul', None]
grid['seasonal_periods'] = [12, 6]
grid['optimized'] = [True, False]
grid['use_boxcon'] = [True, False]
grid['remove_bias'] = [True, False]
all_scores = select_params(test_data=test, train_data=train, grid=grid)
best_scores = all_scores.loc[0, :]

yhat = build_model(train, test, trend=best_scores.loc['paramlist'][0], damped=best_scores.loc['paramlist'][1],
                   seasonal=best_scores.loc['paramlist'][2], seasonal_periods=best_scores.loc['paramlist'][3],
                   optimized=best_scores.loc['paramlist'][4], use_boxcon=best_scores.loc['paramlist'][5],
                   remove_bias=best_scores.loc['paramlist'][6])

# Predict ahead of time using all available data
pred_len = int(len(daily_cases) * .33)
date_range = pd.date_range(start=daily_cases.index[-1] + timedelta(days=1),
                           end=daily_cases.index[-1] + timedelta(days=(pred_len + 1)),
                           periods=pred_len)
df_predict = pd.DataFrame(index=date_range)

yhat_future = build_model(daily_cases, df_predict, trend=best_scores.loc['paramlist'][0],
                          damped=best_scores.loc['paramlist'][1],
                          seasonal=best_scores.loc['paramlist'][2], seasonal_periods=best_scores.loc['paramlist'][3],
                          optimized=best_scores.loc['paramlist'][4], use_boxcon=best_scores.loc['paramlist'][5],
                          remove_bias=best_scores.loc['paramlist'][6])


def plot_forecast(train, pred, test=None):
    fig, (ax_scalar, ax_log) = plt.subplots(2, 1, sharex='all')

    ax_log.set_yscale('log')
    ax_log.plot(train.index, train, label='Train')
    ax_log.plot(pred.index, pred, label='Prediction', color='orange')
    ax_log.set_ylabel('log of Cases')

    ax_scalar.plot(train.index, train, label='Train')
    ax_scalar.plot(pred.index, pred, label='Prediction', color='orange')
    ax_scalar.set_ylabel('Cases')
    ax_scalar.set_xlabel('Date')

    if test is not None:
        ax_scalar.plot(test.index, test, label='Test', color='green')
        ax_log.plot(test.index, test, label='Test', color='green')

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=-45)
    handles, labels = ax_scalar.get_legend_handles_labels()
    ax_scalar.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.7), fancybox=False, shadow=False, ncol=3)

    plt.suptitle('Daily Covid-19 cases prediction')
    fig.tight_layout()
    plt.show()

# plot_forecast(train=daily_cases, pred=yhat_future, test=None)
plot_forecast(train=train, pred=yhat, test=test)

