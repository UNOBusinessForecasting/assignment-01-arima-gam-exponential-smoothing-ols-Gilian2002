[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yOpA_s_M)
# Assignment 1
## Econ 8310 - Business Forecasting

This assignment will make use of the models covered in Lessons 1 to 3. Models include:

- Ordinary Least Squares (OLS) models
- Logistic Regression models
- AutoRegressive Integrated Moving Average (ARIMA) models
- Generalized Additive Models (GAMs)
- Exponential Smoothing models

Your job will be to forecast the number of taxi trips requested during each hour in a week in New York City, utilizing past data about taxi trips in New York City. Your grade will be assigned based on the performance of your code, and will be based primarily on:

- Your code executing without errors
- Storing your models to make predictions on new data
- Making reasonable predictions based on the data provided

The data is available at [https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv](https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv)

To complete this assignment, your code will need to contain the following:

- One valid model. This can be one model of any kind covered in class to this point (see list above). For your model, be sure that you structure the model code as follows:

    - A forecasting algorithm named `model` using the implementation of one of the four models covered in weeks 1 to 3 (don't use other libraries, since I can't keep track of all of them). This model will use the number of trips in an hour as the dependent variable, and may or may not use exogenous variables from the remainder of the dataset.
    - A fitted model named `modelFit`. This should be a model instance capable of generating forecasts by incorporating new data in the same shape as the data used in part (1).
    - A vector of forecasts using the data from the test period named `pred`. You should predict each hour in January of the year following our training data (for 744 total predicted hours).
    
To make predictions, you can use the test data set found at the following link: [https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv](https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv)

While the .ipynb file is a notebook for you to use as you experiment, you must put **all code necessary to complete this assignment into the file called `assignment1.py` found in the file tree**. If your notebook can be run as a script (ie - runs without errors when you restart the kernel and run all cells), then you can simply export your notebook to a .py file and overwrite `assignment1.py`.

**Note:** While all models from weeks 1 to 3 are available to you, they may not all be good fits to the data. I recommend considering the data carefully, then choosing 2-3 models to try. See which models seem to perform best on this data, and implement the best choice for the final submission of the project.

