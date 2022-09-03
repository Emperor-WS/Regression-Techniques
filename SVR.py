#Importing libraries

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1, 1))


#fitting SVR to the dataset
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

#predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#visualising the SVR results(high quality and smoother curve)
plt.scatter(x, y, color="red")
plt.plot(x, regressor.predict(x), color="blue")
plt.title("truth or bluff SVR ")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()
