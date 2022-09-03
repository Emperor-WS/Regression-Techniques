#Importing libraries

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#visualising the linear regression results
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("truth or bluff LR ")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()


#visualising the polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color="red")
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("truth or bluff polynomial ")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()

#predicting a new result with LR
lin_reg.predict([[6.5]])

#predicting a new result with PR
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
