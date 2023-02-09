import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
#from sklearn import linear_model

data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')
data_zscore = data.copy()
data = data.dropna()#Drop any line that includes NULL
data = data.drop_duplicates()#Drop any duplicate data
cols = data.columns
#USe z_score to standardize and remove the outliers
for col in cols:
    data_col = data[col]
    z_score = (data_col - data_col.mean()) / data_col.std()#Standardize the data
    data_zscore[col] = z_score.abs()>2
data_drop = data
for col in cols:
    data_drop = data_drop[data_zscore[col] == False]
df = data_drop.reset_index(drop=True)#df is the dataset we are going to analyse
m = 0
print(df.describe())

for col in df.columns:#Normalization of the data
  df[col]=(df[col].subtract(df[col].mean())).div(df[col].std()).round(3)

print(df.describe())
print(df)

