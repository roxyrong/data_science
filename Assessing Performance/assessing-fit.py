import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

sales = pd.read_csv('Assessing Performance/kc_house_data.csv')

# linear regression
poly_data1 = sales[['sqft_living', 'price']].sort_values(['sqft_living', 'price'])
regr1 = LinearRegression()
regr1.fit(np.array(poly_data1['sqft_living']).reshape(-1, 1), np.array(poly_data1['price']))
predicted = regr1.predict(np.array(poly_data1['sqft_living']).reshape(-1, 1))

plt.plot(poly_data1['sqft_living'], poly_data1['price'], '.',
         poly_data1['sqft_living'], predicted, '-')

# polynomial regression
poly_data2 = sales[['sqft_living', 'price']].sort_values(['sqft_living', 'price'])
X = np.array(poly_data2['sqft_living']).reshape(-1, 1)
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
y = np.array(poly_data2['price'])
regr2 = LinearRegression()
regr2.fit(X, y)
predicted = regr2.predict(X)
plt.plot(poly_data2['sqft_living'], poly_data2['price'], '.',
         poly_data2['sqft_living'], predicted, '-')

# selecting a polynomial degree
train_valid_data, test_data = train_test_split(sales, test_size=0.1)
train_data, valid_data = train_test_split(sales, test_size=0.3)

arr1 = []
arr2 = []
for degree in range(1, 16):
    poly = PolynomialFeatures(degree)
    X_train = np.array(train_data['sqft_living']).reshape(-1, 1)
    X_train = poly.fit_transform(X_train)
    y_train = np.array(train_data['price'])
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    X_valid = np.array(valid_data['sqft_living']).reshape(-1, 1)
    X_valid = poly.fit_transform(X_valid)
    y_valid = np.array(valid_data['price'])
    y_valid_predicted = regr.predict(X_valid)
    residuals_valid = y_valid_predicted - y_valid
    rss1 = sum(residuals_valid * residuals_valid)
    arr1.append(rss1)

    X_test = np.array(test_data['sqft_living']).reshape(-1, 1)
    X_test = poly.fit_transform(X_test)
    y_test = np.array(test_data['price'])
    y_test_predicted = regr.predict(X_test)
    residuals_test = y_test_predicted - y_test
    rss2 = sum(residuals_test * residuals_test)
    arr2.append(rss2)

print(arr1.index(min(arr1)), min(arr1))
print(arr2.index(min(arr2)), min(arr2))

