#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/27 21:55
@Author  : LI Zhe
"""
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd

train = pd.read_csv('../data/after_pre_train_20180928.csv', encoding='gb2312')
y = train[['血糖']]
train.drop(['血糖'], axis=1, inplace=True)
y = y.as_matrix()
X = train.as_matrix()
test = pd.read_csv('../data/after_pre_test.csv', encoding='gb2312')
test = test.as_matrix()

seed = 0
kfold = model_selection.KFold(n_splits=5, random_state=seed)


def bulid_model(model_name):
    model = model_name()
    return model

scoring = 'neg_mean_squared_error'

for model_name in [
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        KNeighborsRegressor,
        DecisionTreeRegressor,
        SVR]:
    if model_name == KNeighborsRegressor:
        results = model_selection.cross_val_score(
            KNeighborsRegressor(n_neighbors=20), X, y, cv=kfold, scoring=scoring)
        print(model_name, abs(results.mean()))
    elif model_name == Lasso:
        results = model_selection.cross_val_score(
            Lasso(max_iter=10000, alpha=0.1), X, y, cv=kfold, scoring=scoring)
        print(model_name, abs(results.mean()))
    else:
        model = bulid_model(model_name)
        results = model_selection.cross_val_score(
            model, X, y, cv=kfold, scoring=scoring)
        print(model_name, abs(results.mean()))

# <class 'sklearn.linear_model.base.LinearRegression'> 2.011333174499831
# <class 'sklearn.linear_model.ridge.Ridge'> 2.007154715469367
# <class 'sklearn.linear_model.coordinate_descent.Lasso'> 2.0314917254948925
# <class 'sklearn.linear_model.coordinate_descent.ElasticNet'> 2.1259007449506084
# <class 'sklearn.neighbors.regression.KNeighborsRegressor'> 2.233808611302681
# <class 'sklearn.tree.tree.DecisionTreeRegressor'> 4.578948918502535
# <class 'sklearn.svm.classes.SVR'> 2.4474301032829855