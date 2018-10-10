#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/29 17:36
@Author  : LI Zhe
"""
import pandas as pd
from sklearn.svm import SVR
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

data_train = pd.read_csv('../data/new_train_feature.csv', low_memory=False, encoding='gbk')

train_x = data_train.iloc[:,:-1]
train_y = data_train.iloc[:,-1]

# Create the RFE object
svr = SVR(kernel="linear", C=1)
rfe = RFE(estimator=svr, n_features_to_select=20, step=1)
rfe.fit(train_x, train_y)
ranking = rfe.ranking_.reshape(train_x[0].shape)

plt.matshow(ranking)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

