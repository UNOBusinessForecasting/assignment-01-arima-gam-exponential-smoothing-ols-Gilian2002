# Import necessary libraries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your dataset (directly from the URL)
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv", parse_dates=['Timestamp'])

# Set 'Timestamp' as the index and ensure the data is sorted
data.set_index('Timestamp', inplace=True)
data = data.sort_index()

# Split the data into training and test sets (let's assume an 80-20 split)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Fit the Exponential Smoothing Model on the training data (additive trend, no seasonality for now)
model = ExponentialSmoothing(train_data['trips'], trend='add').fit()

# Store the fitted model as 'modelFit'
modelFit = model

# Forecast the test period (predict the number of trips for the test period)
test_predictions = modelFit.forecast(steps=len(test_data))

# Combine the test predictions and test actual values into a DataFrame for comparison
test_results = pd.DataFrame({
    'test-valid-predict': test_predictions,
    'test-valid-model': test_data['trips']
})

# Output the comparison table (test predictions vs actual test values)
print(test_results)

# Visualize the fitted values on the training set and the predictions on the test set
fitted_values = modelFit.fittedvalues

plt.figure(figsize=(10,6))
plt.plot(train_data['trips'], label='Train Data (Fitted)', color='blue')
plt.plot(test_data['trips'], label='Test Data (Actual)', color='green')
plt.plot(test_predictions, label='Test Predictions', color='red')
plt.title('Train vs Test Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Number of Trips')
plt.legend()
plt.show()
