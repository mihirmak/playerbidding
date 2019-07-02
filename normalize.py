import pandas as pd
import numpy as np

data = pd.read_csv('unique.csv')
print(data.head())
X = data.drop('No', axis=1).drop('Player', axis=1).drop('Y', axis=1)
#Y = data['']

from sklearn import preprocessing

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

df.to_csv('scaled.csv')