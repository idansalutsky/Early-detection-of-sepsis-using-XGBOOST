from preprocess import *
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import f1_score
import joblib
import sys, os, csv
import pandas as pd
import xgboost as xgb

try1 = ['HR_mean', 'O2Sat_mean', 'Temp_mean', 'MAP_mean',  'Resp_mean', 'Hgb_mean',
       'WBC_mean', 'HospAdmTime_mean', 'ICULOS_mean',
       'ICULOS_std','label_mean']

try2 = ['HR_mean', 'O2Sat_mean', 'Temp_mean', 'MAP_mean',  'Resp_mean','Chloride_mean', 'Lactate_mean', 'Hct_mean', 'Hgb_mean',
       'WBC_mean', 'Platelets_std', 'HospAdmTime_mean', 'ICULOS_mean',
       'ICULOS_std','label_mean']
train_grouped_copy = impute_knn(train_grouped[try2])
test_grouped_copy = impute_knn(test_grouped[try2])

X_train = train_grouped_copy.drop('label_mean', axis=1)
y_train = train_grouped_copy['label_mean']

X_test = test_grouped_copy.drop('label_mean', axis=1)
y_test = test_grouped_copy['label_mean']

# Random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

# Predict the train data
y_pred_train = rf.predict(X_train)

f1_train = f1_score(y_pred_train, y_train)

# Predict the test data
y_pred_test = rf.predict(X_test)

f1_test = f1_score(y_test, y_pred_test)

print("F1 train score:", f1_train)
print("F1 test score:", f1_test)



# Neural network model
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu')) 
model.add(Dense(32, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

y_pred_nn = model.predict(X_test)
y_pred_train = model.predict(X_train)

threshold = 0.55
binary_predictions_test = np.where(y_pred_nn > threshold, 1, 0)
binary_predictions_train = np.where(y_pred_train > threshold, 1, 0)

f1_nn = f1_score(y_test, binary_predictions_test)
f1_train = f1_score(y_train, binary_predictions_train)

print(f"F1 score for neural network train: {f1_train:.4f}")
print(f"F1 score for neural network test: {f1_nn:.4f}")


#xgb model


# create DMatrix for training and test sets
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss', 
    'eta': 0.001, # learning rate
    'subsample': 0.5, # subsampling ratio of training data
    'seed': 42,
    'sampling_method': 'uniform'
}

# train
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=1000)

y_pred_test = xgb_model.predict(dtest)
y_pred_train = xgb_model.predict(dtrain)

range_t = np.arange(0, 1, 0.001)
max = 0
max_ther = 0
for i in range_t:
  binary_predictions = np.where(y_pred_test > i, 1, 0)
  f1 = f1_score(y_test, binary_predictions)
  binary_predictions_train = np.where(y_pred_train > i, 1, 0)

  f1_train =  f1_score(y_train, binary_predictions_train) 
  if max < f1: 
    max = f1
    max_ther = i
    max_train = f1_train

print(f"F1 score for xnm test: {max:.4f}, train {max_train} with threshold {max_ther}")


#saving best model

def make_comp_model(train_grouped, test_grouped):
  df_comp = pd.concat([train_grouped, test_grouped], ignore_index=True)
  comp_grouped_copy = impute_knn(df_comp[try2])

  X_comp = comp_grouped_copy.drop('label_mean', axis=1)
  y_comp = comp_grouped_copy['label_mean']

  dcomp = xgb.DMatrix(X_comp, label=y_comp)
  

  # set XGBoost parameters
  params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss', 
    'eta': 0.001, # learning rate
    'subsample': 0.5, # subsampling ratio of training data
    'seed': 42, # random seed
    'sampling_method': 'uniform'
  }

  # train XGBoost model
  xgb_model = xgb.train(params=params, dtrain=dcomp, num_boost_round=1000)
  joblib.dump(xgb_model, 'comp_xgb_model.joblib')

make_comp_model(train_grouped, test_grouped)


