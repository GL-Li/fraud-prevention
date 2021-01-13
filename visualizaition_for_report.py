#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visualization for the report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% reading data
frauded = pd.read_csv("labels_obf.csv")
frauded["fraud"] = 1
frauded.drop("reportedTime", axis=1, inplace=True)

# transaction data
trans = pd.read_csv("transactions_obf.csv", 
                           parse_dates=['transactionTime'],
                           index_col=0)

#%% Monthly loss in 2017
trans2017 = trans[trans.index.year == 2017]
trans2017['month'] = trans2017.index.month
trans2017.reset_index(inplace=True)

# frauded transations
fraud2017 = pd.merge(trans2017, frauded, on='eventId', how='inner')
fraud2017['loss'] = fraud2017.transactionAmount * fraud2017.fraud


loss_2017 = fraud2017.loss.sum()


month_loss = fraud2017.groupby('month').loss.sum()

plt.bar(range(1, 13), month_loss)


#%% pie plot for 2018 test result
total_loss = 5814.46
prevented = 4079.30
lost = total_loss - prevented

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Lost', 'Prevented'
sizes = [lost, prevented]
explode = (0, 0.05)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#%%
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["1735 Lost",
          "4079 Prevented"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          title="Results",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("Matplotlib bakery: A pie")

plt.show()