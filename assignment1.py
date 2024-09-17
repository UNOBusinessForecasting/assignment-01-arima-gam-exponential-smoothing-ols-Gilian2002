# Import necessary libraries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load your dataset (directly from the URL as specified)
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv", parse_dates=['Timestamp'])

# Set 'Timestamp' as the index and ensure the data is sorted
data.set_index('Timestamp', inplace=True)
data = data.sort_index()

# Splitting the data into training and test sets manually (80% train, 20% test)
train_size = int(len(data) * 0.8)  # 80% of data for training
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Based on your Lecture 3:
# Exponential Smoothing with an Additive Trend (Lecture 3 Content)
# Using a smoothing factor based on Lecture 3 examples
alpha = 0.8  # Adjust as per Lecture 3

# Fit the model on the training data
model = ExponentialSmoothing(train_data['trips'], trend='add').fit(smoothing_level=alpha)

# Store the fitted model
modelFit = model

# Forecast the test period (predict trips for the test period)
test_predictions = modelFit.forecast(steps=len(test_data))

# Create a DataFrame to store the test predictions and actual test values for comparison
test_results = pd.DataFrame({
    'test-valid-predict': test_predictions,
    'test-valid-model': test_data['trips']
})

# Print the test results (comparison of predicted vs actual values)
print("Test Results (Predicted vs Actual):")
print(test_results.head())

# Plot the training data (fitted), test data (actual), and test predictions
fitted_values = modelFit.fittedvalues

plt.figure(figsize=(12,6))
plt.plot(train_data['trips'], label='Train Data (Fitted)', color='blue')
plt.plot(test_data['trips'], label='Test Data (Actual)', color='green')
plt.plot(test_predictions, label='Test Predictions', color='red')
plt.title('Training Data, Test Data, and Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Number of Trips')
plt.legend()
plt.show()

# Calculate the Mean Squared Error manually (based on actual vs predicted values from test set)
mse = ((test_results['test-valid-model'] - test_results['test-valid-predict']) ** 2).mean()
print(f"Mean Squared Error on the Test Data: {mse}")
