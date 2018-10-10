#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/27 21:52
@Author  : LI Zhe
"""
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv('../data/after_pre_train_20180928.csv', encoding='gb2312')
y = train[['血糖']]
# train.drop(['id','血糖'], axis=1, inplace=True)
train.drop(['血糖'], axis=1, inplace=True)
y = y.as_matrix()
# print(y)
X = train.as_matrix()
# print(X)
test = pd.read_csv('../data/after_pre_test.csv', encoding='gb2312')
# test.drop(['id'], axis=1, inplace=True)
test = test.as_matrix()
# print(test)

kf = KFold(n_splits=5)

Linear = LinearRegression()
sum1 = 0

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    Linear.fit(X_train, y_train)
    pre = Linear.predict(X_val)
    mse = mean_squared_error(y_val, pre)
    sum1 += mse
    print("once")
print("avg_mse=", sum1 / 5)  # 2.011333174499831
# Linear.fit(X, y)
# pred = Linear.predict(test)
# np.savetxt('predict_Linear.csv', pred, fmt='%f')
