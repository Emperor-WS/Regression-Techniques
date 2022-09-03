#Importing libraries

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


#Avoiding the dumy variable trap
x = x[:, 1:]

#Spilitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting multiple linear Regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Building the optimal model using Backward Elimination
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
x_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
