# Import necessary libraries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load your dataset (directly from the URL as you mentioned)
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv", parse_dates=['Timestamp'])

# Set 'Timestamp' as the index and ensure the data is sorted
data.set_index('Timestamp', inplace=True)
data = data.sort_index()

# Fit the Exponential Smoothing Model (with additive trend)
model = ExponentialSmoothing(data['trips'], trend='add').fit()

# Store the fitted model as 'modelFit'
modelFit = model

# Print a summary of the model to ensure it worked
print(f"Model Summary:\n{modelFit.summary()}")

# Visualize the fitted values against the actual data
fitted_values = modelFit.fittedvalues

plt.figure(figsize=(10,6))
plt.plot(data['trips'], label='Actual Trips')
plt.plot(fitted_values, label='Fitted Values', color='red')
plt.title('Actual vs Fitted Trips')
plt.xlabel('Timestamp')
plt.ylabel('Number of Trips')
plt.legend()
plt.show()
