import pandas as pd

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

register_matplotlib_converters()

plt.style.use('bmh')
plt.rcParams.update({"axes.facecolor": "white"})

# Load and clean up the data
url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
      r'csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
df = pd.read_csv(url)
# Get daily commulative cases per country
df_per_country = df.groupby('Country/Region').sum(axis=0)
df_per_country = df_per_country.sort_values(df_per_country.columns[-1], ascending=False)
# Drop uneeded columns such as provice, coordinates, etc.
df_per_country = df_per_country.iloc[:, 4:]

# Get all cases per country after the 100th case
df_per_country = df_per_country.T.reset_index()
df_per_country.rename(columns={'index': 'date'}, inplace=True)

dict_cases = {k: list() for k in df_per_country.columns[1:]}
for col in df_per_country.columns[1:]:
    for idx in df_per_country[col].index:
        if df_per_country.loc[idx, col] >= 100:
            dict_cases[col].append(df_per_country.loc[idx, col])
            # Limit the cases to 30 days after the 100th case
            dict_cases[col] = dict_cases[col][0:30]

# Get the 33% growth line
ls_x = list()
x = 100
for i in range(0, 25, 1):
    ls_x.append(x)
    x *= 1.33

# Define Countries/Regions of interest
ls_color_countries = ['China', 'Korea, South', 'Iran', 'Spain', 'UK', 'Singapore', 'Hong Kong', 'Japan', 'Italy',
                      'Netherlands']

# Plot the growth of cases past the 100th case for each country
fig, ax = plt.subplots()
for col in dict_cases.keys():
    # Plot countries of interest with color and annotations
    if col in ls_color_countries:
        ax.plot(dict_cases[col], label=col, linewidth=1.5, marker='o', markersize=3)
        ax.text(ax.lines[-1].get_xdata()[-1]+0.2, ax.lines[-1].get_ydata()[-1]+1,
                col, color=ax.lines[-1].get_color())
    # Plot the rest of the countries in grey and with reduced opacity
    else:
        ax.plot(dict_cases[col], color='grey', linewidth=.8, alpha=0.5)
# Plot the 33% growth line for comparison
ax.plot(ls_x, linestyle='--', color='black', linewidth=1)
ax.text(ax.lines[-1].get_xdata()[-1], ax.lines[-1].get_ydata()[-1],
        '33% growth', color=ax.lines[-1].get_color())
# Use log scale in order not to represent the data in a more readable way
plt.yscale('log')
# Format the y axis in terms of scalar cumulative cases
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_yticks([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 70000])
ax.set_xticks(range(0, 30, 5))
ax.set_xlabel('Number of days since 100th case')
ax.set_ylabel('Cumulative cases')
plt.ylim([100, None])
plt.xlim([0, 35])
plt.tick_params(labelright=True)
plt.show()
