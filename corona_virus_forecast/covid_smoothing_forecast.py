import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use('bmh')
plt.rcParams.update({"axes.facecolor": "white"})

url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
      r'csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
df = pd.read_csv(url)

df = df.iloc[:, 4:]

com_daily_cases = df.sum(axis=0)
com_daily_cases.index = pd.to_datetime(com_daily_cases.index)

fig, ax = plt.subplots()
ax.plot(np.log(com_daily_cases))
ax.set_title('Cumulative daily cases')
ax.set_xlabel('Date')
ax.set_ylabel('log Commulative cases')
plt.xticks(rotation=-45)
plt.show()
plt.close()

daily_cases = com_daily_cases.diff().fillna(com_daily_cases[0]).astype(np.int64)

fig1, ax1 = plt.subplots()
ax1.bar(daily_cases.index, daily_cases)
ax1.set_title('Number of Daily Cases')
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Cases')
plt.xticks(rotation=-45)
plt.show()

# Try to optimize the smoothing




test_data_size = int(0.33 * len(daily_cases))
train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=12).fit()
pred = model.predict(start=test_data.index[0], end=test_data.index[-1])

fig2, ax2 = plt.subplots()
ax2.plot(train_data.index, train_data, label='Train')
ax2.plot(test_data.index, test_data, label='Test')
ax2.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best')
plt.xticks(rotation=-45)
plt.show()



t_params = ['add', 'mul', None]
d_params = [True, False]
s_params = ['add', 'mul', None]
p_params = [12, 6]
b_params = [True, False]
r_params = [True, False]

a = (itertools.product(t_params, d_params, s_params, p_params, b_params, r_params))
list(map(list, a))
