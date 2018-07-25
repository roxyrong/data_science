import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

sales = pd.read_csv('Ridge Regression/kc_house_data.csv')


# polynomial regression
poly_data = sales[['sqft_living', 'price']].sort_values(['sqft_living', 'price'])
X = np.array(poly_data['sqft_living']).reshape(-1, 1)
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
y = np.array(poly_data['price'])
regr1 = LinearRegression()
regr1.fit(X, y)
predicted = regr1.predict(X)
plt.plot(poly_data['sqft_living'], poly_data['price'], '.',
         poly_data['sqft_living'], predicted, '-')

# ridge regression
l2_small_penalty = 1e-5

X = np.array(poly_data['sqft_living']).reshape(-1, 1)
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
y = np.array(poly_data['price'])
regr2 = Ridge(alpha=l2_small_penalty)
regr2.fit(X, y)
predicted = regr2.predict(X)
plt.plot(poly_data['sqft_living'], poly_data['price'], '.',
         poly_data['sqft_living'], predicted, '-')


# select L2 penalty with cross validation

def k_fold_validation(X, y, k, l2_penalty):
    kf = KFold(n_splits=k)
    rss_sum = 0

    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        ridge_regr = Ridge(alpha=l2_penalty)
        ridge_regr.fit(X_train, y_train)
        y_valid_predicted = ridge_regr.predict(X_valid)
        residuals = y_valid - y_valid_predicted
        rss = sum(residuals * residuals)
        rss_sum += rss

    validation_error = rss_sum / k
    return validation_error


poly_data = sales[['sqft_living', 'price']].sort_values(['sqft_living', 'price'])
X = np.array(poly_data['sqft_living']).reshape(-1, 1)
poly = PolynomialFeatures(4)
X = poly.fit_transform(X)
y = np.array(poly_data['price'])

val_dict = {}
for l2 in np.logspace(1, 7, num=13):
    val_error = k_fold_validation(X, y, 10, l2)
    val_dict[l2] = val_error

print(min(val_dict.items(), key=lambda x: x[1]))