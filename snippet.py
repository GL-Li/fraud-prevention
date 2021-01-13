#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:10:47 2020

@author: gl
"""

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
        data[ft] = pd.cut(data.fraud_rate, 
            [-0.1, 0.1 * avg_fraud, 0.5 * avg_fraud, avg_fraud, 2 * avg_fraud, 1.1],
            labels=['A', 'B', 'C', 'D', 'E'])
        data.drop(['fraud_rate'], axis=1, inplace=True)

    return data

features = ['accountNumber', 'merchantId', 'mcc', 'merchantCountry',
            'merchantZip', 'posEntryMode']
aaa = convert_categorical(all_data, features)