!pip install prophet # Only use this line if prophet is not already installed
module = __import__(smoothData)

import pandas as pd
from prophet import Prophet

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data.head()

import statsmodels.formula.api as smf
reg = smf.ols("trips ~ hour", data=data)

reg = reg.fit()

reg.summary()

import pandas as pd
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
print(data)
px.line(data, x="Timestamp", y='trips')

employment = data['trips']
employment.index = data['Timestamp']
employment.index.freq = employment.index.inferred_freq

alpha020 = SimpleExpSmoothing(employment).fit(
                                        smoothing_level=0.2,
                                        optimized=False)

alpha050 = SimpleExpSmoothing(employment).fit(
                                        smoothing_level=0.5,
                                        optimized=False)

alpha080 = SimpleExpSmoothing(employment).fit(
                                        smoothing_level=0.8,
                                        optimized=False)

forecast020 = alpha020.forecast(3)
forecast050 = alpha050.forecast(3)
forecast080 = alpha080.forecast(3)

import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([taxi.values, alpha020.fittedvalues.values,  alpha050.fittedvalues.values,  alpha080.fittedvalues.values]).T
smoothData.columns = ['Truth', 'alpha=0.2', 'alpha=0.5', 'alpha=0.8']
smoothData.index = taxi.index

fig = px.line(smoothData, y = ['Truth', 'alpha=0.2', 'alpha=0.5', 'alpha=0.8'],
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'alpha=0.2': 'red',
                            'alpha=0.5':'green',
                            'alpha=0.8':'purple'}
       )

# Dynamically set x-axis and y-axis ranges
fig.update_xaxes(range=[smoothData.index[-744], forecast020.index[-1]])
fig.update_yaxes(range=[smoothData['Truth'].min() - 1000, smoothData['Truth'].max() + 1000])



# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast020.index, y = forecast020.values, name='Forecast alpha=0.2', line={'color':'red'}))
fig.add_trace(go.Scatter(x=forecast050.index, y = forecast050.values, name='Forecast alpha=0.5', line={'color':'green'}))
fig.add_trace(go.Scatter(x=forecast080.index, y = forecast080.values, name='Forecast alpha=0.8', line={'color':'purple'}))

# Linear trend
model = ExponentialSmoothing(taxi, trend='add', seasonal='add').fit()
# Linear trend with damping
dampedModel = ExponentialSmoothing(taxi, trend='mul', seasonal='add', damped=True, use_boxcox=True).fit(use_brute=True) # Set use_boxcox during initialization

forecast_t = model.forecast(744)
forecast_dt = dampedModel.forecast(744)
import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([taxi.values, model.fittedvalues.values, dampedModel.fittedvalues.values]).T
smoothData.columns = ['Truth', 'Trend', 'Damped Model']
smoothData.index = taxi.index

fig = px.line(smoothData, y = ['Truth', 'Trend', 'Damped Model'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Trend': 'red',
                            'Damped Model': 'green'
                           },
              title='With Seasonality'
       )

# Dynamically set x-axis and y-axis ranges
fig.update_xaxes(range=[smoothData.index[-744], forecast020.index[-1]])
fig.update_yaxes(range=[smoothData['Truth'].min() - 1000, smoothData['Truth'].max() + 1000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast_t.index, y = forecast_t.values, name='Forecast Trend', line={'color':'red'}))
fig.add_trace(go.Scatter(x=forecast_dt.index, y = forecast_dt.values, name='Forecast Damped Model', line={'color':'green'}))
