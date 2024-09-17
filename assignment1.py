import pandas as pd
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing
import plotly.graph_objects as go

# Load the data (from the provided link)
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])  # Ensure the timestamp is in the correct format
print(data.head())

# Set 'Timestamp' as the index
data.set_index('Timestamp', inplace=True)

# Create the target variable 'trips'
taxi = data['trips']

# 1. Define the forecasting algorithm (Exponential Smoothing) and name it `model`
# Exponential Smoothing with additive trend and seasonal component
model = ExponentialSmoothing(taxi, trend='add', seasonal='add', seasonal_periods=24)

# 2. Fit the model and name it `modelFit`
modelFit = model.fit()

# 3. Create a forecast for January (744 hours)
pred = modelFit.forecast(steps=744)

# Visualize the data and forecasts
smoothData = pd.DataFrame({
    'Truth': taxi.values,
    'Fitted': modelFit.fittedvalues
})
smoothData.index = taxi.index

# Create a time index for the forecast (January, 744 hours)
forecast_index = pd.date_range(start=smoothData.index[-1] + pd.Timedelta(hours=1), periods=744, freq='H')

# Plotting the actual data and the forecast
fig = px.line(smoothData, y=['Truth', 'Fitted'], 
              color_discrete_map={'Truth': 'blue', 'Fitted': 'red'},
              title='Taxi Trips with Forecast for January 2019')

# Update x-axis to include the forecast period
fig.update_xaxes(range=[smoothData.index[-744], forecast_index[-1]])

# Dynamically set y-axis range based on the min/max values
fig.update_yaxes(range=[smoothData['Truth'].min() - 1000, smoothData['Truth'].max() + 1000])

# Add the forecast (January 2019) to the plot
fig.add_trace(go.Scatter(x=forecast_index, y=pred.values, name='Forecast', line={'color': 'green'}))

# Show the plot
fig.show()
