import pandas as pd
import statsmodels.api as sm
import plotly.express as px

# Load your dataset (directly from the URL as specified)
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
print(data.head())

# Clean up the data if needed (you can modify this based on any specific cleaning steps required)
data = data[['Timestamp', 'trips']]  # Only keep relevant columns
data['Timestamp'] = pd.to_datetime(data['Timestamp'])  # Convert Timestamp to datetime format

# Plot the raw data
fig = px.scatter(data, x='Timestamp', y='trips', title='Number of Taxi Trips Over Time')
fig.show()

# Set 'Timestamp' as the index and ensure the data is sorted
data.set_index('Timestamp', inplace=True)
data = data.sort_index()

# Let's grab the last 20% of the data for testing and the first 80% for training
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Implement an ARIMA model (order can be adjusted based on data analysis, using ARIMA(1, 0, 0) for simplicity)
arima_model = sm.tsa.ARIMA(train_data['trips'], order=(1, 0, 0)).fit()

# Forecast the test period (predict the number of trips)
pred = arima_model.forecast(steps=len(test_data))

# Create a new DataFrame to compare actual and predicted values
test_results = pd.DataFrame({
    'Timestamp': test_data.index,
    'Actual Trips': test_data['trips'],
    'Predicted Trips': pred
})

# Plot the predictions against actual test data
fig = px.line(test_results, x='Timestamp', y=['Actual Trips', 'Predicted Trips'], title='Actual vs Predicted Taxi Trips')
fig.show()

# Function to return the model, modelFit, and predictions
def get_model_results():
    """
    Returns the model, fitted model, and predictions for test-valid-model.
    """
    return arima_model, arima_model, pred
