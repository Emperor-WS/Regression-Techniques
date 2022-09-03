#Importing libraries

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting the Decision Tree Regression model to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)


#predicting a new result with PR
y_pred = regressor.predict([[6.5]])

#visualising the Decision Tree Regression results(high quality and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("truth or bluff Decision Tree Regression ")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()
