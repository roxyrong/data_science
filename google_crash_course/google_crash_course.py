import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense


pd.options.display.max_rows = 10
pd.options.display.max_columns = 100

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

features = california_housing_dataframe[["total_rooms"]].values
targets = california_housing_dataframe["median_house_value"].values

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(1))
sgd = keras.optimizers.SGD(lr=10)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae', 'accuracy'])
model.fit(x=features, y=targets, epochs=100)
model.evaluate(features, targets)
