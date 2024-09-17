# Import necessary libraries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load your dataset (directly from the URL as specified)
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv", parse_dates=['Timestamp'])

# Set 'Timestamp' as the index and ensure the data is sorted
data.set_index('Timestamp', inplace=True)
data = data.sort_index()

# Splitting the data into training and test sets manually (80% train, 20% test)
train_size = int(len(data) * 0.8)  # 80% of data for training
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Exponential Smoothing model with an additive trend
alpha = 0.8  # Smoothing level based on lecture 3 guidance
model = ExponentialSmoothing(train_data['trips'], trend='add').fit(smoothing_level=alpha)

# Storing the fitted model
modelFit = model

# Forecast the test period (predicting the trips for the test period)
pred = modelFit.forecast(steps=len(test_data))

# Return the required components
def get_model_results():
    """
    This function returns the model, modelFit, and pred as required for validation.
    """
    return model, modelFit, pred

# Display the model summary for verification (you may remove this for the final implementation)
print(modelFit.summary())
