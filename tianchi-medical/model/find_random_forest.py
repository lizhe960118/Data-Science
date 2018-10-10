#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/28 11:00
@Author  : LI Zhe
"""
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.grid_search import GridSearchCV  # Perforing grid search

train_df = pd.read_csv('../data/after_pre_train_20180928.csv', encoding='gb2312')
train_xy, val = train_test_split(train_df, test_size=0.3, random_state=1)

# 训练集
train_y = train_xy['血糖']
train_x = train_xy.drop(['血糖'], axis=1)
# 验证集
val_y = val['血糖']
val_x = val.drop(['血糖'], axis=1)

test = pd.read_csv('../data/after_pre_test.csv', encoding='gb2312')

# cv_params = {'n_estimators': [i for i in range(10, 150, 10)]}
# other_params = {
#     'bootstrap': True,
#     'criterion': 'mse',
#     'max_depth': 2,
#     'max_features': 'auto',
#     'max_leaf_nodes': None,
#     'min_impurity_decrease': 0.0,
#     'min_impurity_split': None,
#     'min_samples_leaf': 1,
#     'min_samples_split': 2,
#     'min_weight_fraction_leaf': 0.0,
#     'n_estimators': 100,
#     'n_jobs': 1,
#     'oob_score': False,
#     'random_state': 0,
#     'verbose': 0,
#     'warm_start': False}
# model = ensemble.RandomForestRegressor(**other_params)
# optimized_randomforest = GridSearchCV(
#     estimator=model,
#     param_grid=cv_params,
#     scoring= 'r2',
#     cv=5,
#     verbose=1
# )
# optimized_randomforest.fit(train_x, train_y)
# # print(optimized_randomforest.grid_scores_, optimized_randomforest.best_params_, optimized_randomforest.best_score_)
# evalute_result = optimized_randomforest.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_randomforest.best_params_))  # {'n_estimators': 130}
# print('最佳模型得分:{0}'.format(optimized_randomforest.best_score_))
# 0.08229216930973242

# cv_params = {'max_depth': [i for i in range(3, 14, 2)], 'min_samples_split':[i for i in range(50, 201, 20)]}
# other_params = {
#     'bootstrap': True,
#     'criterion': 'mse',
#     'max_depth': 2,
#     'max_features': 'auto',
#     'max_leaf_nodes': None,
#     'min_impurity_decrease': 0.0,
#     'min_impurity_split': None,
#     'min_samples_leaf': 1,
#     'min_samples_split': 2,
#     'min_weight_fraction_leaf': 0.0,
#     'n_estimators': 130,
#     'n_jobs': 1,
#     'oob_score': False,
#     'random_state': 0,
#     'verbose': 0,
#     'warm_start': False}
# model = ensemble.RandomForestRegressor(**other_params)
# optimized_randomforest = GridSearchCV(
#     estimator=model,
#     param_grid=cv_params,
#     scoring='r2',
#     cv=5,
#     verbose=1
# )
# optimized_randomforest.fit(train_x, train_y)
# print(optimized_randomforest.grid_scores_, optimized_randomforest.best_params_, optimized_randomforest.best_score_)
# evalute_result = optimized_randomforest.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_randomforest.best_params_)) # {'max_depth': 15, 'min_samples_split': 50}
# print('最佳模型得分:{0}'.format(optimized_randomforest.best_score_))

# cv_params = { 'min_samples_split': [i for i in range(10, 100, 10)], 'min_samples_leaf':[i for i in range(10, 60, 10)]}
# other_params = {
#     'bootstrap': True,
#     'criterion': 'mse',
#     'max_depth': 15,
#     'max_features': 'auto',
#     'max_leaf_nodes': None,
#     'min_impurity_decrease': 0.0,
#     'min_impurity_split': None,
#     'min_samples_leaf': 10,
#     'min_samples_split': 20,
#     'min_weight_fraction_leaf': 0.0,
#     'n_estimators': 130,
#     'n_jobs': 1,
#     'oob_score': False,
#     'random_state': 0,
#     'verbose': 0,
#     'warm_start': False}
# model = ensemble.RandomForestRegressor(**other_params)
# optimized_randomforest = GridSearchCV(
#     estimator=model,
#     param_grid=cv_params,
#     scoring='r2',
#     cv=5,
#     verbose=1
# )
# optimized_randomforest.fit(train_x, train_y)
# print(optimized_randomforest.grid_scores_, optimized_randomforest.best_params_, optimized_randomforest.best_score_)
# evalute_result = optimized_randomforest.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_randomforest.best_params_)) # {'min_samples_split': 10, 'min_samples_leaf': 20}
# print('最佳模型得分:{0}'.format(optimized_randomforest.best_score_))

# cv_params = {'max_features':[i for i in range(3,11,2)]}
# other_params = {
#     'bootstrap': True,
#     'criterion': 'mse',
#     'max_depth': 15,
#     'max_features': 'auto',
#     'max_leaf_nodes': None,
#     'min_impurity_decrease': 0.0,
#     'min_impurity_split': None,
#     'min_samples_leaf': 10,
#     'min_samples_split': 20,
#     'min_weight_fraction_leaf': 0.0,
#     'n_estimators': 130,
#     'n_jobs': 1,
#     'oob_score': False,
#     'random_state': 0,
#     'verbose': 0,
#     'warm_start': False}
# model = ensemble.RandomForestRegressor(**other_params)
# optimized_randomforest = GridSearchCV(
#     estimator=model,
#     param_grid=cv_params,
#     scoring='r2',
#     cv=5,
#     verbose=1
# )
# optimized_randomforest.fit(train_x, train_y)
# print(optimized_randomforest.grid_scores_, optimized_randomforest.best_params_, optimized_randomforest.best_score_)
# evalute_result = optimized_randomforest.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_randomforest.best_params_)) # {'max_features': 9}
# print('最佳模型得分:{0}'.format(optimized_randomforest.best_score_)) #0.14964599211203777

model = ensemble.RandomForestRegressor(
    bootstrap=True,
    criterion='mse',
    max_depth= 15,
    max_features= 9,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    min_samples_leaf=20,
    min_samples_split=10,
    min_weight_fraction_leaf=0.0,
    n_estimators=130,
    n_jobs=1,
    oob_score=False,
    random_state=0,
    verbose=0,
    warm_start=False)
model.fit(train_x, train_y)
preds = model.predict(val_x)
print(metrics.mean_squared_error(val_y, preds))  # 1.6834881268876962
