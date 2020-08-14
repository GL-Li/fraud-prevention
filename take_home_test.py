#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:39:02 2020

@author: gl
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix,accuracy_score

#%% manually create features
labeled = pd.read_csv("labels_obf.csv")
labeled["fraud"] = 1
labeled.drop("reportedTime", axis=1, inplace=True)

transactions = pd.read_csv("transactions_obf.csv", 
                           parse_dates=['transactionTime'],
                           index_col=0)
transactions['month'] = transactions.index.month.astype(str)
transactions['weekday'] = transactions.index.weekday.astype(str)
transactions['hour'] = transactions.index.hour.astype(str)


all_data = pd.merge(transactions, labeled, on='eventId', how='left')
all_data.drop(["eventId"], axis=1, inplace=True)
all_data.fraud.fillna(0, inplace=True)
all_data.fraud = all_data.fraud.astype(int)

# convert integer to string
for ft in ["mcc", "merchantCountry", "posEntryMode"]:
    all_data[ft] = all_data[ft].astype(str)
    
#%% represent categorical features with log odd
def convert_categorical(labeled_data, features, new_data=None):
    avg_fraud = all_data.fraud.mean()
    
    if new_data is None:
        data = labeled_data.copy()
    else:
        data = new_data
        
    for ft in features:
        fraud_prob = (labeled_data.groupby(ft, as_index=False).fraud.mean().
                      rename(columns={'fraud':'fraud_rate'}))
        data = pd.merge(data, fraud_prob, on=ft, how='left')
        # in case of unseen categories, fill with avg_fraud
        data['fraud_rate'] = data['fraud_rate'].fillna(avg_fraud)
        data[ft] = np.log((data.fraud_rate + 1e-6)/(1 - data.fraud_rate + 1e-6))
        data.drop(['fraud_rate'], axis=1, inplace=True)

    return data

features = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry',
            'merchantZip', 'posEntryMode']
aaa = convert_categorical(all_data, features)


#%% build logistic model
"""
To predict future, use 2017 full year data as training data and 2018 Janury as
test data
"""
train = all_data[transactions.index.year == 2017]
test = all_data[transactions.index.year == 2018]

y_train = train.fraud
y_test = test.fraud
X_train = train.drop(['fraud'], axis=1)
X_test = test.drop(['fraud'], axis=1)

X_train = convert_categorical(train, features, X_train)
X_test = convert_categorical(train, features, X_test)

col_trans = ColumnTransformer(
        transformers=[('cats', StandardScaler(), [0, 1, 2, 3, 4, 5]),
                      ('moneys', PowerTransformer(), [6, 7]),
                      ('date', OneHotEncoder(drop='first'), [8, 9, 10])])
X_train = col_trans.fit_transform(X_train)
X_test = col_trans.transform(X_test)

logit = LogisticRegression(random_state=9876, class_weight={0:1, 1:1000})
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)

confusion_matrix(y_test, y_pred)
