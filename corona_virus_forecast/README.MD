# Covid19 Data Analysis & Forecast
> An analysis of the growth of the confirmed covid19 cases and timeseries forecast

I have used the [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv)
data in order to make a visual analysis of the growth of the confirmed covid19 cases,
as well as make a prediction about that growth.

<br>

For the analysis of the growth I have graphed the cumulative cases per country, for 30 days after the 100th case in the
respective country. I have highlighted some countries of interest and have compared their growth paths with the 33% growth
rate. The resulting figure gives interesting insight about the development of the pandemic. Namely, in many western 
countries the growth of the confirmed cases has been comparable to a daily growth of 33%, which could be interpreted as
a sign that many other countries might soon be facing the challenges оf Italy. However, in countries such as Singapore,
Japan, Hong Kong, and South Korea, the growth has been reduced by the use of a set of rapid and strict measures, such as
mass testing, social distancing, and tracking of potential virus carriers, among others.

<br>
The growth of confirmed covid-19 cases per country (with the countries of interest being color-highlighted), after the 100th case. The plotted series are log-transformed:

![Covid19_growth_analysis](https://github.com/dargiroff/timeseries_analysis/blob/master/corona_virus_forecast/covid19_analysis.png)

<br>

For the forecast of the daily covid19 cases, I have used [Holt-Winter's exponential smoothing technique](https://otexts.com/fpp2/holt-winters.html). <br>
Via a grid search, I have found the best hyperparams for the exponential smoothing and have backtest the forecast on
a training set that constitutes 33% of the available data.
After the backtest I have used all available data to make a prediction in the future.

<br>
Backtest of the Holt-Winter's exponential smoothing prediction of daily covid19 cases:

![Covid19_forecast_backtest](https://github.com/dargiroff/timeseries_analysis/blob/master/corona_virus_forecast/covid19_forecast_backtest.png)

<br>
Forecast of the daily covid19 cases using Holt-Winter's exponential smoothing:

![Covid19_forecast_backtest](https://github.com/dargiroff/timeseries_analysis/blob/master/corona_virus_forecast/covid19_forecast.png)



