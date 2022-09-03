#Importing libraries

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting the Random Forest Regression model to the dataset
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x, y)
#predicting a new result with Random Forest Regression
y_pred = regressor.predict([[6.5]])

#visualising the Random Forest Regression results(high quality and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("truth or bluff Random Forest Regression ")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()
