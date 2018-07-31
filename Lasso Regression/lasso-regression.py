import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, normalize
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", 20)
sales = pd.read_csv('Lasso Regression/kc_house_data.csv')

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
feature_matrix = sales[features]
target = sales['price']


# Lasso Regression
coef = pd.DataFrame(index=np.logspace(1, 7, num=13), columns=features + ['rss'])

for l1 in np.logspace(1, 7, num=13):
    regr = Lasso(alpha=l1, normalize=True)
    regr.fit(feature_matrix, target)
    predicted = regr.predict(feature_matrix)
    rss = sum((predicted - target) * (predicted - target))
    coef.loc[l1] = np.append(regr.coef_, rss)

print(coef.apply(lambda x: sum(x.values != 0), axis=1)) # print feature number


# Implement Coordinate Descent
features = ['sqft_living', 'bedrooms']
feature_matrix = sales[features].values
poly = PolynomialFeatures(1)
feature_matrix = poly.fit_transform(feature_matrix)
weights = np.ones(feature_matrix.shape[1])
target = np.array(sales['price'])
l1_penalty = 1

# norms = np.sqrt(np.sum(feature_matrix ** 2, axis=0))
# normlized_features = feature_matrix / norms


def coordinate_descent(feature_matrix, target, weights, l1_penalty):
    rho = [0 for i in range(feature_matrix.shape[1])]
    for i in range(feature_matrix.shape[1]):
        predicted = np.dot(feature_matrix, weights)
        rss = sum((predicted - target) * (predicted - target))
        print(weights, rss)
        rho[i] = sum(weights[i] * (target - predicted + weights[i] * feature_matrix[:, i]))
        if i == 0:
            weights[i] = rho[i]
        elif rho[i] < -l1_penalty / 2:
            weights[i] = rho[i] + l1_penalty / 2
        elif rho[i] > l1_penalty / 2.:
            weights[i] = rho[i] - l1_penalty / 2
        else:
            weights[i] = 0
