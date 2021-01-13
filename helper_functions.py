#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:56:31 2020

@author: gl
"""

def process_data(data):
    """To clean and manipulate transaction data"""
    data['month'] = data.index.month.astype(str)
    data['weekday'] = data.index.weekday.astype(str)
    data['hour'] = data.index.hour.astype(str)
    # 19 discrete availableCash, so turn to categorical feature
    data['availableCash'] = data.availableCash.astype(str)