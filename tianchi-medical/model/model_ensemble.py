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
# from sklearn.cross_validation import train_test_split, KFold
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from catboost import CatBoostRegressor


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

y_train = pd.DataFrame(y_train)
x_train = pd.DataFrame(x_train)

# print(y_train)
# print(x_train)

test_data = pd.DataFrame(df_test)

SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)


def get_out_flod(model, x_train, y_train, test_data):
    train_preds_model = np.zeros(x_train.shape[0],)
    # train_preds_model 用来保存每次交叉验证每个模型的预测，n个模型拼接即为下层的训练数据
    test_preds_model_k = np.zeros((test_data.shape[0], NFOLDS))
    # test_preds_model 用来保存每次对于测试集的预测，多次取平均即为第一层对测试集的预测
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        print(model.__class__, '第{}次训练...'.format(i))

        x_train_temp = x_train.iloc[train_index]
        y_train_temp = y_train.iloc[train_index]
        x_test_temp = x_train.iloc[test_index]

        model.fit(x_train_temp, y_train_temp)

        train_preds_model[test_index] = model.predict(x_test_temp).ravel()

        test_preds_model_k[:,i] = model.predict(test_data).ravel()
    test_preds_model = test_preds_model_k.mean(axis=1)
    return train_preds_model, test_preds_model

def get_model_byname(model_name):
    if model_name == 'Lasso':
        model = Lasso(max_iter=10000, alpha=0.1)
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(
            learning_rate=0.1,
            n_estimators=84,
            max_depth=3,
            min_child_weight=6,
            seed=0,
            subsample=0.9,
            colsample_bytree=0.6,
            gamma=0.5,
            reg_alpha=2,
            reg_lambda=3)
    elif model_name == 'gdbt':
        params = {'n_estimators': 600, 'max_depth': 2, 'min_samples_split': 4,
                  'learning_rate': 0.025, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif model_name == 'randomforest':
        model = RandomForestRegressor(
            bootstrap=True,
            criterion='mse',
            max_depth=15,
            max_features=9,
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
    elif model_name == "catboost":
        model = CatBoostRegressor(
            iterations=20 * 40,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=1,
            random_seed=0)
    else:
        model = model_name()
    return model


model_layer_first = [LinearRegression,
                     Ridge,
                     Lasso,
                     ElasticNet,
                     'gdbt',
                     'randomforest'
                     ]

train_data_layer_first = np.zeros(
    (train_data.shape[0], len(model_layer_first)))
test_data_layer_first = np.zeros((test_data.shape[0], len(model_layer_first)))

for i, model_name in enumerate(model_layer_first):
    model = get_model_byname(model_name)
    train_preds_model, test_preds_model = get_out_flod(
        model, x_train, y_train, test_data)
    train_data_layer_first[:, i] = train_preds_model
    test_data_layer_first[:, i] = test_preds_model

train_data_layer_first = pd.DataFrame(data=train_data_layer_first[0:,0:],    # values
                                      index= [i for i in range(train_data.shape[0])],   # 1st column as index
                                      columns= [i + 1for i in range(len(model_layer_first))])  # 1st row as the column names
test_data_layer_first = pd.DataFrame(data=test_data_layer_first[0:, 0:],
                                     index = [i for i in range(test_data.shape[0])],
                                     columns= [i + 1 for i in range(len(model_layer_first))])
# model_layer_second = [
#     'xgboost'
# ]
# for model_name in model_layer_second:
#     model = get_model_byname(model_name)
#     train_data_layer_second, test_data_layer_second = get_out_flod(
#         model, train_data_layer_first, y_train, test_data_layer_first)

model = get_model_byname('xgboost')
train_data_layer_second, test_data_layer_second = get_out_flod(
    model, train_data_layer_first, y_train, test_data_layer_first)
# model_second = get_model_byname("xgboost")
# train_data_layer_second = np.zeros(train_data_layer_first.shape[0])
# # train_preds_model 用来保存每次交叉验证每个模型的预测，n个模型拼接即为下层的训练数据
# test_data_layer_second_k = np.zeros((test_data_layer_first.shape[0], 5))
# # test_preds_model 用来保存每次对于测试集的预测，多次取平均即为第一层对测试集的预测
# for i, (train_index, test_index) in enumerate(kf):
#     print('第{}次训练...'.format(i))
#     x_train_temp = train_data_layer_first[train_index]
#     y_train_temp = y_train.iloc[train_index]
#     x_test_temp = train_data_layer_first[test_index]
#     model_second.fit(x_train_temp, y_train_temp)
#     train_data_layer_second[test_index] += model_second.predict(x_test_temp)
#     test_data_layer_second_k[:, i] = model_second.predict(test_data_layer_first)
# test_data_layer_second = test_data_layer_second_k.mean(axis=1)

# model_layer_last = [
#     'catboost'
# ]
#
# for model_name in model_layer_last:
#     model = get_model_byname(model_name)
#     catboost_train = model.fit(train_data_layer_second, y_train)
#     prediction = model.predict(test_data_layer_second)

train_data_layer_second = pd.DataFrame(data=train_data_layer_second[0:,],    # values
                                      index= [i for i in range(train_data.shape[0])],   # 1st column as index
                                      columns= ["血糖"])  # 1st row as the column names
test_data_layer_second = pd.DataFrame(data=test_data_layer_second[0:,],
                                     index = [i for i in range(test_data.shape[0])],
                                     columns= ["血糖"])

# print(train_data_layer_second.shape[0],train_data_layer_second.shape[1],test_data_layer_second.shape[0]) #5641, 1, 1000
# print(y_train.shape) (5641, 1)

model = get_model_byname('catboost')
catboost_train = model.fit(train_data_layer_second, y_train)

print(
    '线下mse： {}'.format(
        metrics.mean_squared_error(
            y_train,
            train_data_layer_second) *
        0.5))

prediction = model.predict(test_data_layer_second)

prediction = pd.DataFrame(data=prediction[:,],
                        index = [i for i in range(test_data.shape[0])],
                        columns= ["血糖"])
# print(prediction.shape[0], prediction.shape[1])

print(
    '实际mse： {}'.format(
        metrics.mean_squared_error(
            df_test_A_answer,
            prediction) *
        0.5))

# 线下得分： 0.935115832011848
# 实际验证得分： 0.8758847159326207

# submission_lgb = pd.DataFrame({'pred': online_test_preds_lgb})
# submission_lgb['pred'].to_csv('../data/0930_lgb.csv', header=None, index=False)
