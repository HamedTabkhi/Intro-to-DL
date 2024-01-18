import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm.notebook import tqdm

df_train = pd.read_csv('./datasets/house-train.csv')
df_test = pd.read_csv('./datasets/house-test.csv')

# Check data type
pd.options.display.max_rows=90
df_dtype = pd.DataFrame(df_train.dtypes,columns=['dtype'])
print(df_dtype.value_counts())
print(df_dtype)

# Features used to modeling
usefull_cols = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF'
                , 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces'
                ,'LotFrontage','WoodDeckSF','OpenPorchSF'
                ,'ExterQual','Neighborhood','MSZoning'
                ,'Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition',]
df_train_prepro = df_train[usefull_cols].copy()
df_test_prepro = df_test[usefull_cols].copy()

# Remove Nulls 
## GarageArea in test data
df_test_prepro['GarageArea'] = df_test_prepro['GarageArea'].fillna(df_train_prepro['GarageArea'].mean())
## TotalBsmtSF in test data
df_test_prepro['TotalBsmtSF'] = df_test_prepro['TotalBsmtSF'].fillna(df_train_prepro['TotalBsmtSF'].mean())

# One-hot encoding
df_train_prepro = pd.get_dummies(df_train_prepro,columns=['Neighborhood','MSZoning','Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition'])
df_test_prepro = pd.get_dummies(df_test_prepro,columns=['Neighborhood','MSZoning','Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition'])
#One-hot encoding: convert categorical data variables into a form that could be provided to machine learning.
#It creates binary (0 or 1) columns for each category in the original data.
#pd.get_dummies(): Tconverts categorical variable(s) into dummy/indicator variables.

df_train_prepro = df_train_prepro.replace({True: 1, False: 0})
df_test_prepro = df_test_prepro.replace({True: 1, False: 0})

# Convert all columns to numeric (float) and handle NaN values
df_train_prepro = df_train_prepro.apply(pd.to_numeric, errors='coerce').fillna(0)
df_test_prepro = df_test_prepro.apply(pd.to_numeric, errors='coerce').fillna(0)

# Save the DataFrame to a CSV file
output_file = 'temp/housing_df_train_prepro.csv'
df_train_prepro.to_csv(output_file, index=False)
print(f'DataFrame saved to {output_file}')