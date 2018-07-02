import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

PATH = "C:/Users/ckoja/.kaggle/competitions/house-prices-advanced-regression-techniques/"

dataframe = pd.read_csv(PATH + "train.csv")
testing_set = pd.read_csv(PATH + "test.csv")

print('The original shape of our features is:', dataframe.shape)

print('Removing features (columns) with more than 70% missing...')
dataframe= dataframe[dataframe.columns[dataframe.isnull().mean() < 0.70]]

print('The shape of our features now is:', dataframe.shape)

# HAVE TO REMOVE NANS - BETTER WAY TO DO THIS
dataframe.dropna(inplace=True)
print('The shape of our features after dropping null valued rows is:', dataframe.shape)

# Separate target variable for testing
labels = np.array(dataframe['SalePrice'])
dataframe = dataframe.drop('SalePrice', axis = 1)
feature_list = list(dataframe.columns)

# Concatenate training and test and do one hot encoding
combined_data = pd.concat((dataframe, testing_set)).reset_index(drop=True)

# VARIABLE ENCODING, IMPUTATION, FEATURE ENGINEERING GOES HERE
# Use combined data to edit features so that both train and test are ready

# Split data back to training and testing
train = combined_data[:dataframe.shape[0]]
test = combined_data[dataframe.shape[0]:]

# RF

# Extract numeric variables for baseline model
numeric_variables = list(dataframe.dtypes[dataframe.dtypes != "object"].index)

# Fitting baseline model as a start
print("fitting baseline model on just numerical values...")
modelBaseline = RandomForestRegressor(n_jobs=-1, oob_score=True, random_state=42)
modelBaseline.fit(train[numeric_variables], labels)
print(" ")

# RMSE? KFOLD?
# ----------------------------------------------------------------------------------------------------------------------

# Fitting tuned model after parametric testing based on gridsearch
print("fitting tuned model on full dataset...")
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=15, max_depth=10, n_jobs=-1, max_features=0.2, oob_score=True, random_state=42)
#Once NaNs are removed
#model.fit(train, labels)
print(" ")

# RMSE? KFOLD?
# ----------------------------------------------------------------------------------------------------------------------

# Using GridSearch to analysing ideal model parameters
print("Using GridSearch to extract ideal parameter values")
param_grid = {
    "n_estimators": [2, 10, 25, 50, 75, 100, 125, 150],
    "max_features": ['sqrt', 0.1, 0.2, 0.3, 0.4, 0.5],
    "max_depth": [1, 5, 10, 15, 20, 25],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10, 15, 20, 25]
}

grid_model = RandomForestRegressor(n_jobs=-1, oob_score=True, random_state=42)

CV_model = GridSearchCV(estimator=grid_model, param_grid=param_grid, cv=5)

#print(CV_model.best_params_)
# ----------------------------------------------------------------------------------------------------------------------