#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/30 12:17
@Author  : LI Zhe
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.cross_validation import train_test_split,KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb


# 载入数据
train_data = pd.read_csv(
    '../data/after_pre_train_20180929.csv',
    low_memory=False,
    encoding='gbk')

df_test = pd.read_csv(
    '../data/after_pre_test.csv',
    encoding='gbk')

df_test_A_answer = pd.read_csv(
    '../data/d_answer_a_20180128.csv', header=-1)
#
# df_test['血糖'] = df_test_A_answer[0]

y_train = train_data['血糖']
train_data.drop(['血糖'], axis=1, inplace=True)
# y_train = y_train.as_matrix()
# x_train = train_data.as_matrix()
x_train = train_data

test_data = df_test.as_matrix()

kf = KFold(len(x_train), n_folds=5, shuffle=True, random_state=520)
# poisson regression
lgb_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'num_leaves': 12,
    'colsample_bytree': 0.6,
    'max_depth': 6,
    'min_data': 5,
    'min_hessian': 1,
    'verbose': -1}

# def get_out_flod(model, x_train, y_train, test_data):
train_preds_lgb = np.zeros(train_data.shape[0])
test_preds_lgb = np.zeros((test_data.shape[0], 5))

def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = metrics.mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)

for i, (train_index, test_index) in enumerate(kf):
    print('\n')
    print('第{}次训练...'.format(i))

    x_train_temp = x_train.iloc[train_index]
    y_train_temp = y_train.iloc[train_index]

    x_test_temp = x_train.iloc[test_index]
    y_test_temp = y_train.iloc[test_index]

    print('lightgbm')
    lgb_train = lgb.Dataset(x_train_temp,y_train_temp)
    lgb_val = lgb.Dataset(x_test_temp, y_test_temp)
    gbm = lgb.train(lgb_params,
                    lgb_train,
                    num_boost_round=20000,
                    valid_sets=lgb_val,
                    verbose_eval=500,
                    feval=evalerror,
                    early_stopping_rounds=200)
    train_preds_lgb[test_index] += gbm.predict(x_test_temp)
    test_preds_lgb[:, i] = gbm.predict(test_data)
    print('\n')

print(
    '线下得分（越低越好）： {}'.format(
        metrics.mean_squared_error(
            y_train,
            train_preds_lgb) *
        0.5))

online_test_preds_lgb = test_preds_lgb.mean(axis=1)

print(
    '实际验证得分(误差越低越好)： {}'.format(
        metrics.mean_squared_error(
            df_test_A_answer,
            online_test_preds_lgb
            ) *
        0.5))

# 线下得分： 0.8775758446779379
# 实际验证得分： 0.845334593066526

# submission_lgb = pd.DataFrame({'pred': online_test_preds_lgb})
# submission_lgb['pred'].to_csv('../data/0930_lgb.csv', header=None, index=False)