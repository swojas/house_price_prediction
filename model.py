import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

dataset = pd.read_csv('train.csv')
y = dataset.iloc[:, -1].values

dataset.drop(['TARGET(PRICE_IN_LACS)'], axis = 1, inplace = True)

dataset.drop(['ADDRESS'], axis=1, inplace=True)

dataset = pd.get_dummies(dataset, drop_first = True)

X = dataset.iloc[:, :]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)



import pickle

pickle.dump(regressor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
