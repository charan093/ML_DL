# -*- coding: utf-8 -*-


#Regression - House sale price prediction
import os
import pandas as pd
import numpy as np
import seaborn as sns

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data")

house_train = pd.read_csv("house_train.csv")
house_train.shape
house_train.info()

house_test = pd.read_csv("house_test.csv")
house_test.shape
house_test.info()

#Concatenate train and test data and we will seperate before applying the model
house_data = pd.concat([house_train, house_test],ignore_index=True)
house_data.drop(["Id","SalePrice"], 1, inplace=True)
house_data.shape
house_data.info()

#convert numerical columns to categorical type              
house_data['MSSubClass'] = house_data['MSSubClass'].astype('category')
house_data.shape
#convert categorical columns to numeric type
ordinal_features1 = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
#ordinal_features1 = [col for col in house_train if 'TA' in list(house_train[col])]
quality_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for feature in ordinal_features1:
    null_idx = house_data[feature].isnull()
    house_data.loc[null_idx, feature] = None 
    house_data[feature] = house_data[feature].map(quality_dict)

#10,20, 30, 50, 2000
#Log of above number
np.log(10) # 2.3025850929940459
np.log(20) #2.9957322735539909
np.log(30) #3.4011973816621555
np.log(50) #3.912023005428146
np.log(2000) #7.6009024595420822

#handle missing data columns
#See, for how many rows the data is missing!
total_missing = house_data.isnull().sum()

#for just dry run, go ahead and delete where the data is missing.
#This is just for a dry run to understand the next steps.
#in reality, we have impute these values
to_delete = total_missing[total_missing>0]
house_data.drop(list(to_delete.index), axis=1, inplace=True)
house_data.shape

#Numerical columns.. include=['number']
numeric_cols = house_data.select_dtypes(include=['number']).columns
#Categorical columns .. exclude = ['number'])
cat_cols = house_data.select_dtypes(exclude = ['number']).columns

#Automated to get all category columns and use one hot encoding instead of writing each and every column
house_data1 = pd.get_dummies(house_data, columns=cat_cols)
house_data1.shape
house_data1.describe
#Plot my sale price
x = house_train['SalePrice']
sns.distplot(x, kde=False)

#splitting train data as conctenated in the begining
house_train1 = house_data1[:house_train.shape[0]]
#splitting test data as conctenated in the begining
house_test1 = house_data1[house_train.shape[0]:]

#Smooting the SalePrice. As the sale price has outlier/noise data
house_train['log_sale_price'] = np.log(house_train['SalePrice'])

#See how the data looks like with log values.
x=house_train['log_sale_price']
sns.distplot(x,kde=False)
