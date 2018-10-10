#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/30 22:18
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from catboost import CatBoostRegressor

# 载入数据
train_data = pd.read_csv(
    '../data/after_pre_train_20180928.csv',
    low_memory=False,
    encoding='gbk')

df_test = pd.read_csv(
    '../data/after_pre_test.csv',
    encoding='gbk')

df_test_A_answer = pd.read_csv(
    '../data/d_answer_a_20180128.csv', header=-1)
#
# df_test['血糖'] = df_test_A_answer[0]

# y_train = pd.DataFrame(train_data['血糖'])
# train_data.drop(['血糖'], axis=1, inplace=True)
# x_train = pd.DataFrame(train_data)
# print(y_train)
# print(x_train)

test_data = pd.DataFrame(df_test)

def probability_pridect(category_other, category_blood_sugar, x_test):
    category_model = LogisticRegression()
    category_model.fit(category_other, category_blood_sugar)

    # 预测出来x_test probability是0还是1
    pred_test_number_blood_sugar = category_model.predict_proba(x_test)
    pred_test_categroy_blood_sugar = pd.Series(np.argmax(pred_test_number_blood_sugar,axis=1))

    # 将测试集的数据加入到训练集进行二次训练
    concat_blood_sugar = category_blood_sugar.append(pred_test_categroy_blood_sugar, ignore_index=True)
    concat_other = category_other.append(x_test, ignore_index=True)
    category_model.fit(concat_other, concat_blood_sugar)

    # pred_train_probability = category_model.predict(category_other)
    pred_test_probability = category_model.predict_proba(x_test)

    return concat_blood_sugar, pred_test_probability


def probability_columns(train_data, test_data, threshold):
    label_list = train_data.columns.tolist()
    if label_list.__contains__('血糖'):
        label_list.remove('血糖')

    # 增加 血糖类别 标签
    train_data['categroy_blood_sugar'] = ((train_data['血糖'] >= threshold) + 1)
    # print(train_data.shape[0]) #5641
    # print(test_data.shape[0]) #1000
    # 根据阈值大小，构造两类训练集
    bigger_train_data = train_data[train_data['血糖'] >= threshold]
    bigger_train_data_y = bigger_train_data["血糖"]
    bigger_train_data.drop(['血糖'], axis = 1, inplace=True)
    bigger_train_data_x = bigger_train_data

    smaller_train_data = train_data[train_data['血糖'] < threshold]
    smaller_train_data_y = smaller_train_data['血糖']
    smaller_train_data.drop(['血糖'], axis=1, inplace=True)
    smaller_train_data_x = smaller_train_data

    # print(len(bigger_train_data), len(smaller_train_data))

    categroy_other = train_data[label_list]
    categroy_blood_sugar = train_data['categroy_blood_sugar']

    concat_blood_suger,pred_test_probability = probability_pridect(categroy_other, categroy_blood_sugar, test_data)
    # print(concat_blood_suger) #5641

    # 提取出来test数据
    categroy_blood_sugar_test = pd.DataFrame(concat_blood_suger[len(train_data):].reset_index())
    # print(categroy_blood_sugar_test)
    categroy_blood_sugar_test.drop(['index'], axis=1, inplace=True)
    # print(categroy_blood_sugar_test)
    test_data = pd.concat([test_data, categroy_blood_sugar_test], axis=1)

    # test_data['categroy_blood_sugar'] = categroy_blood_sugar_test
    # test_add_categroy = pd.concat([test_x, categroy_blood_sugar_test], axis= 1)

    return bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data, train_data, pred_test_probability

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


def model_fit(model, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data):
    model_for_biggger = model
    model_for_smaller = model

    model_for_biggger.fit(bigger_train_data_x, bigger_train_data_y)
    model_for_smaller.fit(smaller_train_data_x, smaller_train_data_y)

    prediction_by_bigger_model = model_for_biggger.predict(test_data)
    prediction_by_smaller_model = model_for_smaller.predict(test_data)

    return model_for_biggger, model_for_smaller, prediction_by_bigger_model, prediction_by_smaller_model

threshold = 6.5
bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data, train_data, pred_test_probability = probability_columns(train_data, test_data, threshold)

model_layer_first = [LinearRegression,
                         Ridge,
                         Lasso,
                         ElasticNet,
                         'gdbt',
                         'randomforest'
                        ]

test_data_layer_first = np.zeros((test_data.shape[0], len(model_layer_first)))

for i, model_name in enumerate(model_layer_first):
    model = get_model_byname(model_name)
    model_for_biggger, model_for_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(model, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
    test_data_layer_first[:, i] = np.array([prediction_by_smaller_model[i] * pred_test_probability[i][0] + prediction_by_bigger_model[i] * pred_test_probability[i][1] for i in range(len(test_data))])

prediction = test_data_layer_first.mean(axis=1)


print(
    '实际mse： {}'.format(
        metrics.mean_squared_error(
            df_test_A_answer,
            prediction) *
        0.5))

# 实际mse： 1.9599628395065083