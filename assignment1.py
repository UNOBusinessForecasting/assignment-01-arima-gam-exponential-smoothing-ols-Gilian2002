!pip install prophet # Only use this line if prophet is not already installed

import pandas as pd
from prophet import Prophet

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data.head()
