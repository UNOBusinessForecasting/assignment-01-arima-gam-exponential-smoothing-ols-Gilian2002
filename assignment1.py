import pandas as pd
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
import plotly.graph_objects as go

# Load the dataset
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
print(data.head())

# Create the target variable 'trips'
taxi = data['trips']

# Fit three Simple Exponential Smoothing models with different smoothing levels (0.2, 0.5, and 0.8)
alpha020 = SimpleExpSmoothing(taxi).fit(smoothing_level=0.2, optimized=False)
alpha050 = SimpleExpSmoothing(taxi).fit(smoothing_level=0.5, optimized=False)
alpha080 = SimpleExpSmoothing(taxi).fit(smoothing_level=0.8, optimized=False)

# Forecast for 744 hours (for January 2019)
forecast020 = alpha020.forecast(steps=744)
forecast050 = alpha050.forecast(steps=744)
forecast080 = alpha080.forecast(steps=744)

# Create a DataFrame with the actual data and fitted values
smoothData = pd.DataFrame([taxi.values, alpha020.fittedvalues.values, alpha050.fittedvalues.values, alpha080.fittedvalues.values]).T
smoothData.columns = ['Truth', 'alpha=0.2', 'alpha=0.5', 'alpha=0.8']
smoothData.index = taxi.index

# Plot the actual values and fitted values for different smoothing levels
fig = px.line(smoothData, y=['Truth', 'alpha=0.2', 'alpha=0.5', 'alpha=0.8'],
              x=smoothData.index,
              color_discrete_map={"Truth": 'blue', 'alpha=0.2': 'red', 'alpha=0.5': 'green', 'alpha=0.8': 'purple'},
              title='Simple Exponential Smoothing with Different Alpha Values')

# Set the x-axis and y-axis dynamically based on the forecast period
fig.update_xaxes(range=[smoothData.index[-744], forecast020.index[-1]])
fig.update_yaxes(range=[smoothData['Truth'].min() - 1000, smoothData['Truth'].max() + 1000])

# Add the forecast data to the plot
fig.add_trace(go.Scatter(x=forecast020.index, y=forecast020.values, name='Forecast alpha=0.2', line={'color': 'red'}))
fig.add_trace(go.Scatter(x=forecast050.index, y=forecast050.values, name='Forecast alpha=0.5', line={'color': 'green'}))
fig.add_trace(go.Scatter(x=forecast080.index, y=forecast080.values, name='Forecast alpha=0.8', line={'color': 'purple'}))

# Show the plot
fig.show()

# Now we will define a model and fit it with Exponential Smoothing for trend and seasonality

# 1. Define the forecasting algorithm (Exponential Smoothing) and name it `model`
model = ExponentialSmoothing(taxi, trend='add', seasonal='add', seasonal_periods=24)

# 2. Fit the model and name it `modelFit`
modelFit = model.fit()

# 3. Forecast for January (744 hours) and name it `pred`
pred = modelFit.forecast(steps=744)

# Create a new DataFrame for the historical data and the fitted values
smoothData = pd.DataFrame({
    'Truth': taxi.values,
    'Trend': modelFit.fittedvalues
})
smoothData.index = taxi.index

# Create the forecast index for January
forecast_index = pd.date_range(start=smoothData.index[-1] + pd.Timedelta(hours=1), periods=744, freq='H')

# Plot the actual data, fitted values, and forecast
fig = px.line(smoothData, y=['Truth', 'Trend'], 
              color_discrete_map={'Truth': 'blue', 'Trend': 'red'},
              title='Taxi Trips with Forecast for January 2019')

# Update the x-axis to include the forecast period
fig.update_xaxes(range=[smoothData.index[-744], forecast_index[-1]])

# Dynamically set the y-axis based on the truth values
fig.update_yaxes(range=[smoothData['Truth'].min() - 1000, smoothData['Truth'].max() + 1000])

# Add the forecast (January 2019) to the plot
fig.add_trace(go.Scatter(x=forecast_index, y=pred.values, name='Forecast', line={'color': 'green'}))

# Show the updated plot
fig.show()
