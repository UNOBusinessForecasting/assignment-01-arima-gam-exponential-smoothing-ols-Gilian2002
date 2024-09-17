from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Load the training data
train_data = pd.read_csv('/mnt/data/assignment_data_train.csv')
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
train_data.set_index('Timestamp', inplace=True)

# Use the ARIMA model
model = ARIMA(train_data['trips'], order=(1, 1, 1))  # ARIMA model with p=1, d=1, q=1 as a starting example
model_fit = model.fit()

# Display a summary of the ARIMA model
model_fit.summary()
