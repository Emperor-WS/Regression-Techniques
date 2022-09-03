#Importing libraries

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Spilitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


#Fitting SLR to the trainning set

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results

y_pred = regressor.predict(x_test)

#visuallising the trainning set results
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vs Experience (training set)")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

#visuallising the test set results
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vs Experience (test set)")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()
