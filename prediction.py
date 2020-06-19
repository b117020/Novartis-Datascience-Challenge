# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 02:01:54 2020

@author: Devdarshan
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

df = pd.read_csv('Train.csv')
df.isnull().any()
df.fillna(df.mean(), inplace=True)
df['X_12'] = [round(item) for item in df['X_12']]

df = df.values
x = df[:,1:17]
y = df[:,17]
y= np.array(y).astype(float)
print(x[0:4,0])


encoder = LabelEncoder()
encoder.fit(x[:,0])
x[:,0]= encoder.transform(x[:,0])
x= np.array(x).astype(float)
#x = preprocessing.scale(x)
x[:,0] = preprocessing.scale(x[:,0])

x[:,11] = preprocessing.scale(x[:,11])
x[:,13:15] = preprocessing.scale(x[:,13:15])
'''
def create_baseline():
    model = Sequential()
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
	

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=10, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, x, y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

model = Sequential()
model.add(Dense(32, input_dim=16, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=150)

test_data = pd.read_csv('Test.csv')
test_data.isnull().any()
test_data.fillna(test_data.mean(), inplace=True)
test_data['X_12'] = [round(item) for item in test_data['X_12']]

td = test_data.values
test = td[:,1:17]
encoder.fit(test[:,0])

test[:,0]= encoder.transform(test[:,0])

test = np.array(test).astype(float)
test[:,0] = preprocessing.scale(test[:,0])

test[:,11] = preprocessing.scale(test[:,11])
test[:,13:15] = preprocessing.scale(test[:,13:15])
scaled_test = preprocessing.scale(test)

predictions = model.predict_classes(test).tolist()
outputs = []
for item in predictions:
    outputs.append(item[0])

dataset = pd.DataFrame({'INCIDENT_ID': td[:, 0], 'MULTIPLE_OFFENSE': outputs})
dataset.to_csv('outputs.csv', index=False)