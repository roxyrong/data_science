import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sales = pd.read_csv('Multiple Regression/kc_house_data.csv')
train, test = train_test_split(sales, test_size=0.2)
features = ['sqft_living', 'bedrooms', 'bathrooms']

# multiple linear regression
regr = LinearRegression()
regr.fit(train[features], train['price'])

predicted = regr.predict(test[features])
expected = test['price']

mse = np.mean((predicted-expected)**2)

print(regr.intercept_, regr.coef_, mse)
print(regr.score(train[features], train['price']))


# gradient descent method
train['constant'] = 1
features = features + ['constant']
features_df = train[features]
features_matrix = features_df.values
target = np.array(train['price'])


converged = False
step_size = 1e-11
weights = np.zeros(4)
tolerance = 1500000000
steps_count = 0
while (not converged) and (steps_count < 1000000):
    steps_count += 1
    predicted = np.dot(features_matrix, weights)
    errors = predicted - target
    gradient_sum_squares = 0
    for i in range(len(weights)):
        derivative = 2 * np.dot(errors, features_matrix[:, i])
        gradient_sum_squares += (derivative ** 2)
        weights[i] -= (step_size * derivative)
    gradient_magnitude = sqrt(gradient_sum_squares)
    print(weights, gradient_magnitude)
    if gradient_magnitude < tolerance:
        converged = True

print(weights)
