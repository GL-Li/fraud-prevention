#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tasks and methods:
- We will identify ~400 high risk transactions each month for the team to
  review.
- The model will predict the probability of a transaction being fraud. 
  - Use logistic regression as it produces well-defined probability.
  - Use 2017 data as training data and 2018 Jan data as test data
- Metrics: as much fraud loss prevented as possible. 
  - For this purpose we will use fraud_probability * transactionAmount to rank
    each transaction.
  - Pick the top 400 transaction for review.

Retults:
- We set January 2018 as the test data, which has 39 frauds reported with a
  total loss of 5814.46 GBP.
- Using the model and team review, we identified 10 frauds and prevented loss
  of 4079.30 GBP.
- We prevented 70% fraud loss in terms of value.
"""
print("Loading packages ...................................................\n")

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
#from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score

pd.options.mode.chained_assignment = None  # default='warn'


#%% define functions to clean data and extract features

def keep_levels(data, features, lab_data=None, n=20):
    """
    For categorical features, keep levels that have at least n fraud reported
    and all other levels groupd as "others"
    
    Parameters
    ----------
    data: data frame whose categorical features are processed
    features: string, list of selected features to be process
    lab_data: labeled transaction data to generate fraud count of each levels
    n: integer, threshold of reported fraud cases. 
    
    Return
    ------
    data frame
    """
    if lab_data is None:
        lab_data = data
    
    for ft in features:
        total_fraud = lab_data.groupby(ft).fraud.sum()
        level_keep = total_fraud.index[total_fraud >= n].tolist()
        data.loc[~data[ft].isin(level_keep), ft] = 'others'
        
    return data

def clean_trans(data):
    """
    To clean transaction data
    
    Parameters
    ----------
    data: data frame, in which 'transactionTime' was set as index
    
    Return
    ------
    data frame
    """
    data = change_types(data)
    data = bin_trans_amount(data)
    data = bin_cash(data)
    
    return data


def change_types(data):
    """
    To change data types of features
    
    Parameters
    ----------
    data: data frame, in which 'transactionTime' was set as index
    
    Return
    ------
    data frame
    """
    
    trans['month'] = trans.index.month.astype(str)
    trans['weekday'] = trans.index.weekday.astype(str)
    trans['hour'] = trans.index.hour.astype(str)

    # convert integer to string
    for ft in ["mcc", "merchantCountry", "posEntryMode"]:
        trans[ft] = trans[ft].astype(str)
        
    return data


def bin_trans_amount(data, all_data=None, n=10):
    """
    To cut 'transactionAmount' into n bins with roughly equal size.
    
    Parameters
    ----------
    data: data frame, transaction data in which transactionAmount is binned
    all_data: data frame, all transactions used to find bins
    n: integer, number of bins
    
    Return
    ------
    data frame of binned data
    """
    if all_data is None:
        all_data = data.copy()
    
    # use pd.qcut to first find the bins with equal number of samples and then
    # set ends to infinity to allow new transactionAmount out of known range
    _, bins = pd.qcut(all_data.transactionAmount, n, retbins=True)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    data['transactionAmount'] = pd.cut(data.transactionAmount, bins)
    
    return data


def bin_cash(data):
    """
    To cut avalibleCash into bins
    
    Parameters
    ----------
    data: data frame, transaction data
    
    Return
    ------
    data frame
    """
    # the bins are made manually with the help of histogram
    bins = [-np.inf, 1500, 2500, 4500, 7500, 8500, 10500, np.inf]
    data['availableCash'] = pd.cut(data.availableCash, bins)
    
    return data

#%% manually create features
print("Reading and cleaning data ..........................................\n")

# fraud labeled data
frauded = pd.read_csv("labels_obf.csv")
frauded["fraud"] = 1
frauded.drop("reportedTime", axis=1, inplace=True)

# transaction data
trans = pd.read_csv("transactions_obf.csv", 
                           parse_dates=['transactionTime'],
                           index_col=0)
original_trans = trans.copy()
trans = clean_trans(trans)


# labeled transaction data
lab_trans = pd.merge(trans, frauded, on='eventId', how='left')
lab_trans.drop(["eventId"], axis=1, inplace=True)
lab_trans.fraud.fillna(0, inplace=True)
lab_trans.fraud = lab_trans.fraud.astype(int)

features = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry',
           'merchantZip', 'posEntryMode']
lab_trans = keep_levels(lab_trans, features)


#%% prepare train and test data
print("Preprocessing train and test data ..................................\n")

train = lab_trans[trans.index.year == 2017]
test = lab_trans[trans.index.year == 2018]

y_train = train.fraud
y_test = test.fraud
X_train = train.drop(['fraud'], axis=1)
X_test = test.drop(['fraud'], axis=1)

col_trans = ColumnTransformer(
        transformers=[('cats', OneHotEncoder(drop='first'), list(range(10)))])
col_trans.fit(pd.concat([X_train, X_test], axis=0))
X_train = col_trans.transform(X_train)
X_test = col_trans.transform(X_test)


#%% train the model
print("Training the logistic regression model .............................\n")

# grid search we set the class_weight to the class ratio and tune a couple of C
model_params = {"C": [0.1, 1, 10]}

logit = LogisticRegression(random_state=9876, class_weight={0:1, 1:135})

grid = GridSearchCV(logit, model_params, scoring="roc_auc", cv=10, 
                    n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print("Evaluate Final model on test data ..................................\n")
y_prob = grid.predict_proba(X_test)
auc = roc_auc_score(y_test, y_prob[:, 1])
print(f"The final model has a AUC score of {auc} on test data.\n")


#%% Evaluate loss prevention with the model
print(f"Evaluating fraud prevention of the model ..........................\n")

# Find the top 400 transaction for human team to review and evaluation how much
# loss prevented by the model
orig_test = original_trans[original_trans.index.year == 2018]
orig_test.reset_index(inplace=True)
orig_test['fraud'] = list(test.fraud)
orig_test['prob'] = y_prob[:, 1]
orig_test['possible_loss'] = orig_test.transactionAmount * orig_test.prob
orig_test.sort_values('possible_loss', ascending=False, inplace=True)

top_400 = orig_test.iloc[0:400, :]

top_eventId = list(top_400.eventId)
print(f"Here are the eventID of 400 transactions to review:\n{top_eventId}\n")

# fraud summary in 2018 Jan
fraud_times = sum(test.fraud)
total_loss = round(sum(orig_test.transactionAmount * orig_test.fraud), 2)
print(f"There are {fraud_times} frauds reported in 2018 January.")
print(f"The total loss is {total_loss} GBP.\n")

# fraud and loss identified by model
identified_fraud = sum(top_400.fraud)
prevent_loss = round(sum(top_400.transactionAmount * top_400.fraud), 2)
print(f"The model identified {identified_fraud} frauds.")
print(f"The total prevented loss is {prevent_loss} GBP.\n")

pct = round((prevent_loss / total_loss) * 100, 2)
print(f"We prevented {pct}% fraud loss.\n")

