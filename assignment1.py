import statsmodels.formula.api as smf
import pandas as pd

# Load training data
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

# Preview data
print(data.head())

# Fit the OLS (Ordinary Least Squares) model using 'hour' as the independent variable
model = smf.ols("trips ~ hour", data=data)
model_fit = model.fit()

# Display the summary of the fitted model
print(model_fit.summary())

# Load test data and only select the 'hour' column
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_data = test_data[['hour']]

# Preview the test data
print(test_data.head())

# Use the fitted model to predict the 'trips' based on the 'hour' from the test data
predictions = model_fit.predict(test_data)

# Output predictions
print(predictions)

# Optional: Save predictions to CSV file for further use
test_data['Predicted_Trips'] = predictions
test_data.to_csv('predicted_trips.csv', index=False)

print("Predictions saved to 'predicted_trips.csv'")
