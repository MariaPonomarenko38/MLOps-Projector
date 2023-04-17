import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [2, 2], [3, 3]])
y = np.array([2, 4, 6])

model = LinearRegression()
model.fit(X, y)

print(model.coef_)