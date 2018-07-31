import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", 20)
sales = pd.read_csv('Kernel Regression/kc_house_data.csv')
features = ['sqft_living', 'bedrooms']
feature_matrix = sales[features].values
poly = PolynomialFeatures(1)
X = poly.fit_transform(feature_matrix)
y = np.array(sales['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
rss = sum((pred - y_test) * (pred - y_test))

plt.plot(y_test, pred, '.')