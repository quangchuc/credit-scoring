# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:13:36 2020

@author: quang-chuc.nguyen
"""
# Import libraries
import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets, metrics, model_selection, svm
from visualisation import plot_roc_curve

train_brut = pd.read_csv(r'C:\Users\quang-chuc.nguyen\Documents\Python\1 Credit scoring\train.csv')
y = train_brut['label']
train_brut = train_brut.drop(['label'], axis=1)
test_brut = pd.read_csv(r'C:\Users\quang-chuc.nguyen\Documents\Python\1 Credit scoring\test.csv')
data = pd.concat(objs=[train_brut,test_brut], axis=0)

num_features = ['age_source1', 'age_source2', 'FIELD_3',
                'FIELD_11', 'FIELD_22', 'FIELD_51',
                'FIELD_52', 'FIELD_53', 'FIELD_55', 
                'FIELD_56', 'FIELD_57']
cat_features = ['province', 'district', 'maCv',
                'FIELD_7', 'FIELD_9', 
                'FIELD_13', 'FIELD_39', 'FIELD_50']
bool_features = ['FIELD_1', 'FIELD_2', 'FIELD_8', 
                 'FIELD_10', 'FIELD_12', 'FIELD_14',
                 'FIELD_15', 'FIELD_18', 'FIELD_19',
                 'FIELD_20', 'FIELD_23', 'FIELD_25', 
                 'FIELD_26', 'FIELD_27', 'FIELD_28', 
                 'FIELD_29', 'FIELD_30', 'FIELD_31',
                 'FIELD_32', 'FIELD_33', 'FIELD_34',
                 'FIELD_36', 'FIELD_37', 'FIELD_38',
                 'FIELD_42', 'FIELD_46', 'FIELD_47', 
                 'FIELD_48', 'FIELD_49']
ordinal_features = [col for col in train_brut.columns if col not in 
                    num_features+cat_features+bool_features+['id', 'label']]

table = data[data['age_source1']==data['age_source2']]
from function import check_missing_data
a = check_missing_data(data[ordinal_features])


data.FIELD_45.value_counts()
data.FIELD_45.isnull().sum()


# Numerical Features 
for col in ['FIELD_3']: 
    data[col] = data[col].fillna(-1)
for col in ['FIELD_22']: 
    data[col] = data[col].fillna(0)
data.age_source1.fillna(data.age_source2, inplace=True)
data.age_source2.fillna(data.age_source1, inplace=True)
for col in ['age_source1', 'age_source2']: 
    data[col]=data[col].fillna(data[col].median())
for col in ['FIELD_55']: 
    data[col] = data[col].fillna(data[col].mean())
for col in ['FIELD_11', 'FIELD_51', 'FIELD_52', 'FIELD_53', 'FIELD_56', 'FIELD_57']: 
    data[col] = data[col].fillna(data[col].value_counts().index[0])
data['FIELD_11'] = data['FIELD_11'].replace('None', data['FIELD_11'].value_counts().index[0])
data['FIELD_11'] = data['FIELD_11'].astype(float)

# Ordinal Features
for col in ['FIELD_4', 'FIELD_5', 'FIELD_6', 
            'FIELD_21', 'FIELD_41', 'FIELD_43', 
            'FIELD_44', 'FIELD_45', 'FIELD_54']:
    data[col] = data[col].fillna(data[col].value_counts().index[0])
data['FIELD_41'] = data['FIELD_41'].map({'None':-1, 'I':1, 'II':2, 'III':3, 'IV':4, 'V':5})
data['FIELD_43'] = data['FIELD_43'].map({'None':-1, 'A':1, 'B':2, 'C':3, 'D':4, '0': 0, '5':5})
data['FIELD_44'] = data['FIELD_44'].map({'None':-1, 'One':1, 'Two':2})
data['FIELD_45'] = data['FIELD_45'].replace('None', '-1')
data['FIELD_45'] = data['FIELD_45'].astype(int)
# Categorical Features



# Boolean Features

# Choose train and test data
train_net = data.iloc[0:30000, :]
test_net = data.iloc[30000:, :]

# Choose features to pass through the model 
chosen_features = num_features + ['FIELD_4', 'FIELD_5', 'FIELD_6', 'FIELD_21', 
                                  'FIELD_41', 'FIELD_43', 'FIELD_44', 'FIELD_45', 'FIELD_54']
X = train_net[chosen_features]
plt.hist(X.iloc[:,10])
X_different = X[X['age_source1']!=X['age_source2']]
X_df = pd.DataFrame()
X_df['age'] = X[X['age_source1']==X['age_source2']]
X_df['age'].isnull().sum()
# Build the model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
grid_params_rfc = {'class_weight': ['balanced', None], 
                   'max_depth': [4, 6, 8], 
                   'max_features': ['auto', 6, 8], 
                   'n_estimators': [10, 50, 100, 200], 
                   'random_state': [10, 20, 30, 40]}
rfc = RandomForestClassifier(grid_params_rfc)
print('Parameters currently in use:\n')
pprint(rfc.get_params())

grid_search = GridSearchCV(estimator = rfc, param_grid = grid_params_rfc, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X, y)
grid_search.best_params_

model = grid_search.best_estimator_
pprint(model.get_params())
plt.plot(model.feature_importances_)
# Fit the model to the whole dataset 
model.fit(X, y)
predict_train = model.predict_proba(X)
predict_train = predict_train[:,1]
auc =  roc_auc_score(y, predict_train)
print('\nAucroc_score on the whole train dataset : ', auc)
fpr, tpr, thresholds = metrics.roc_curve(y, predict_train)
plot_roc_curve(fpr, tpr)

plt.hist(predict_train, bins=30)
stats.describe(predict_train)

# Test and export submission file
X_test = test_net[chosen_features]
predict_test = model.predict_proba(X_test)
predict_test = predict_test[:,1]
submission = pd.DataFrame({'id': test_brut['id'], 'label': predict_test})
submission.to_csv('submission.csv', index=False)
stats.describe(predict_test)