import pandas as pd
import numpy as np

PATH = "C:/Users/ckoja/.kaggle/competitions/house-prices-advanced-regression-techniques/"

dataframe = pd.read_csv(PATH + "train.csv")
testing_set = pd.read_csv(PATH + "test.csv")

print('The original shape of our features is:', dataframe.shape)

print('Removing features (columns) with more than 70% missing...')
dataframe= dataframe[dataframe.columns[dataframe.isnull().mean() < 0.70]]

print('The shape of our features now is:', dataframe.shape)

#dataframe.dropna(inplace=True)
print('The shape of our features after dropping null valued rows is:', dataframe.shape)

# Separate target variable for testing
labels = np.array(dataframe['SalePrice'])
dataframe = dataframe.drop('SalePrice', axis = 1)
feature_list = list(dataframe.columns)

# Concatenate training and test and do one hot encoding
combined_data = pd.concat((dataframe, testing_set)).reset_index(drop=True)

# VARIABLE ENCODING, IMPUTATION, FEATURE ENGINEERING GOES HERE
# Use combined data to edit features so that both train and test are ready
#

# Split data back to training and testing
train = combined_data[:dataframe.shape[0]]
test = combined_data[dataframe.shape[0]:]

