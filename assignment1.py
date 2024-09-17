#%%
import statsmodels.formula.api as smf
import pandas as pd

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

#%%
data.head()

#Select the data, - assign 1
model = smf.ols("trips ~ hour", data=data)

modelFit = model.fit()

modelFit.summary()
print(modelFit.summary())

#%%
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_data = test_data[['hour']]
test_data.head(10)

#%%
pred = modelFit.predict(test_data)

print(pred)



