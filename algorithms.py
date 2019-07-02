from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
data = pd.read_csv('scaled.csv')
data = data.drop('No', axis=1)

X = data.drop('RankPercent', axis=1)
Y = data['RankPercent']


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

###########################################################
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
scr = reg.score(X_test, y_test)
#print(scr)
# 0.3600836547592847
#print(reg.coef_)

from sklearn.metrics import mean_squared_error

score = mean_squared_error(y_test, reg.predict(X_test))
print("linear regression: ", score)
#############################################################


#####################################################################
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

n = nn.fit(X_train, y_train)

#y_pred = nn.predict(X_train)
scr = nn.score(X_test, y_test)
#print(scr)
score = mean_squared_error(y_test, nn.predict(X_test))
print("MLPregressor: ", score)
#0.3702685509025747
##########################################################################

##########################################
from sklearn.svm import SVR

clf = SVR(degree=5, gamma='scale', C=1.0, epsilon=0.0001)
clf.fit(X_train, y_train)
scr = clf.score(X_test, y_test)
score = mean_squared_error(y_test, clf.predict(X_test))
print('SVMR: ', score)
#0.3132268514598787
#############################################


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=25,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, input_dim=25,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, input_dim=25,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, input_dim=25,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model,
                           epochs=200, batch_size=5, verbose = 0)


estimator.fit(X_train, y_train)
scr = estimator.score(X_test, y_test)
print(scr)
y_pred = estimator.predict(X_test)

score = mean_squared_error(y_test, estimator.predict(X_test))
print("Deep NN: ", score)
#r2 = r2_score(y_test, y_pred)
#print(r2)
# 0.3445108367171906
# 0.3612083730671892 //200 epochs, no activation
# 0.36304349263413993 //200 epochs , activation = relu
# 0.36586558665547375 // another dense layer
# print(y_pred)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)

#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)

'''
(venv) hemal@linux:~/Workspaces/ML-Workspace/FCI_Project$ python algorithms.py 
Using TensorFlow backend.
linear regression:  74.72006429903175
MLPregressor:  73.53082120442174
SVMR:  80.1913159421847
Deep NN:  74.00393574611624
(venv) hemal@linux:~/Workspaces/ML-Workspace/FCI_Project$