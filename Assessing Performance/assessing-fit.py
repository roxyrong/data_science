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

