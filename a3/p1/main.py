import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

k = 5

x = np.load('X.npy')
y = np.load('y.npy')

linear_regressor = LinearRegression()
lasso_regressor = Lasso()
ridge_regressor = Ridge()

linear_scores = []
lasso_scores = []
ridge_scores = []

kfold = KFold(k, True, 1)
for train_idx, test_idx in kfold.split(x):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train each regressor using each fold
    linear_regressor.fit(x_train, y_train)
    linear_pred = mean_absolute_percentage_error(y_test, linear_regressor.predict(x_test))
    linear_scores.append(linear_pred)

    lasso_regressor.fit(x_train, y_train)
    lasso_pred = mean_absolute_percentage_error(y_test, lasso_regressor.predict(x_test))
    lasso_scores.append(lasso_pred)

    ridge_regressor.fit(x_train, y_train)
    ridge_pred = mean_absolute_percentage_error(y_test, ridge_regressor.predict(x_test))
    ridge_scores.append(ridge_pred)

print('LINEAR REGRESSION MAPE:', np.average(linear_scores))
print('LASSO REGRESSION MAPE:', np.average(lasso_scores))
print('RIDGE REGRESSION MAPE:', np.average(ridge_scores))
