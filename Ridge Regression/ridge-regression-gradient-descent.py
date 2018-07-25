import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

sales = pd.read_csv('Ridge Regression/kc_house_data.csv')

data = sales[['sqft_living', 'price']].sort_values(['sqft_living', 'price'])
feature_matrix = np.array(data['sqft_living']).reshape(-1, 1)
poly = PolynomialFeatures(2)
feature_matrix = poly.fit_transform(feature_matrix)
target = np.array(data['price'])

weights = np.zeros(3)
converged = False
step_size = 1e-16
tolerance = 1500000000
steps_count = 0
max_iteration = 100
l2_penalty = 1e-3

while max_iteration > 0:
    predictions = np.dot(feature_matrix, weights)
    errors = target - predictions
    for i in np.arange(len(weights)):
        if i == 0:
            derivative = 2 * np.dot(errors, feature_matrix[:, i])
        else:
            derivative = 2 * np.dot(errors, feature_matrix[:, i]) + 2 * (l2_penalty * weights[i])
        weights[i] = weights[i] - step_size * derivative
    max_iteration -= 1
    print(weights)

