import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sales = pd.read_csv('Multiple Regression/kc_house_data.csv')
train, test = train_test_split(sales, test_size=0.2)

example_features = ['sqft_living', 'bedrooms', 'bathrooms']
regr = LinearRegression()
regr.fit(train[example_features], train['price'])

predicted = regr.predict(test[example_features])
expected = test['price']

mse = np.mean((predicted-expected)**2)

print(regr.intercept_, regr.coef_, mse)
print(regr.score(train[example_features], train['price']))
