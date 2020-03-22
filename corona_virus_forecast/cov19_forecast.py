import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use('bmh')
plt.rcParams.update({"axes.facecolor": "white"})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
#       r'csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
# df = pd.read_csv(url)

df = pd.read_csv(r'/Users/dimitar/IdeaProjects/timeseries_analysis/data/time_series_19-covid-Confirmed.csv')
df = df.iloc[:, 4:]

com_daily_cases = df.sum(axis=0)
com_daily_cases.index = pd.to_datetime(com_daily_cases.index)

fig, ax = plt.subplots()
ax.plot(com_daily_cases)
ax.set_title('Cumulative daily cases')
ax.set_xlabel('Date')
ax.set_ylabel('Commulative cases')
plt.xticks(rotation=-45)
plt.show()
plt.close()

daily_cases = com_daily_cases.diff().fillna(com_daily_cases[0]).astype(np.int64)

fig1, ax1 = plt.subplots()
ax1.plot(daily_cases)
ax1.set_title('Cumulative daily cases')
ax1.set_xlabel('Date')
ax1.set_ylabel('Commulative cases')
plt.xticks(rotation=-45)
plt.show()

# Split the data into test and training
test_data_size = int(0.33 * len(daily_cases))
train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

# Scale the data for the neural network
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))


def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


class CoronaVirusPredictor(nn.Module):

    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    # Override the forward method
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


def train_model(
        model,
        train_data,
        train_labels,
        test_data=None,
        test_labels=None
):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 60

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.float(), y_train)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    return model.eval(), train_hist, test_hist


model = CoronaVirusPredictor(
    n_features=1,
    n_hidden=512,
    seq_len=seq_length,
    n_layers=2
)
model, train_hist, test_hist = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)

fig2, ax2 = plt.subplots()
ax2.plot(train_hist, label="Training loss")
ax2.plot(test_hist, label="Test loss")
plt.ylim((0, 5))
fig2.legend()
plt.show()

with torch.no_grad():
    test_seq = X_test[:1]
    preds = []
    for _ in range(len(X_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

        true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()

true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

fig3, ax3 = plt.subplots()
ax3.plot(daily_cases.index[:len(train_data)], scaler.inverse_transform(train_data).flatten(),
         label='Historical Daily Cases')
ax3.set_xlabel('Date')
ax3.set_ylabel('Daily Cases')
plt.xticks(rotation=-45)
fig3.legend()
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(daily_cases.index[len(train_data):len(train_data) + len(true_cases)], true_cases, label='Real Daily Cases')
ax4.plot(daily_cases.index[len(train_data):len(train_data) + len(true_cases)], predicted_cases,
         label='Predicted Daily Cases')
ax4.set_xlabel('Date')
ax4.set_ylabel('Daily Cases')
plt.xticks(rotation=-45)
fig4.legend()
plt.show()

scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))
all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))

X_all, y_all = create_sequences(all_data, seq_length)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model = CoronaVirusPredictor(n_features=1, n_hidden=512, seq_len=seq_length, n_layers=2)
model, train_hist, _ = train_model(model, X_all, y_all)

DAYS_TO_PREDICT = 14

with torch.no_grad():
    test_seq = X_all[:1]
    preds = []
    for _ in range(DAYS_TO_PREDICT):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()
predicted_index = pd.date_range(start=daily_cases.index[-1], periods=DAYS_TO_PREDICT + 1, closed='right')
predicted_cases = pd.Series(data=predicted_cases, index=predicted_index)

fig5, ax5 = plt.subplots()
ax5.plot(predicted_cases, label='Predicted Daily Cases')
ax5.set_xlabel('Date')
ax5.set_ylabel('Daily Cases')
plt.xticks(rotation=-45)
fig5.legend()
plt.show()

fig6, ax6 = plt.subplots()
ax6.plot(daily_cases, label='Historical Daily Cases')
ax6.plot(predicted_cases, label='Predicted Daily Cases')
ax6.set_xlabel('Date')
ax6.set_ylabel('Daily Cases')
plt.xticks(rotation=-45)
fig6.legend()
plt.show()