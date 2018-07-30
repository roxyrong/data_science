import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
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
