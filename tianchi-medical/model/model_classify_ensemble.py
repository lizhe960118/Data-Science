#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/2 0:06
@Author  : LI Zhe
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import model_selection
# from sklearn.cross_validation import train_test_split, KFold
from sklearn.model_selection import KFold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import xgboost as xgb

from mlxtend.regressor import StackingRegressor

# 载入数据
train_data = pd.read_csv(
    '../data/after_pre_train_20180930.csv',
    low_memory=False,
    encoding='gbk')

df_test = pd.read_csv(
    '../data/after_pre_test_20180930.csv',
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
    pred_test_categroy_blood_sugar = pd.Series(
        np.argmax(pred_test_number_blood_sugar, axis=1))

    # 将测试集的数据加入到训练集进行二次训练
    concat_blood_sugar = category_blood_sugar.append(
        pred_test_categroy_blood_sugar, ignore_index=True)
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
    bigger_train_data.drop(['血糖'], axis=1, inplace=True)
    bigger_train_data_x = bigger_train_data
    # bigger_train_data_x = bigger_train_data.drop(['血糖'], axis=1, inplace=True)

    smaller_train_data = train_data[train_data['血糖'] < threshold]
    smaller_train_data_y = smaller_train_data['血糖']
    smaller_train_data.drop(['血糖'], axis=1, inplace=True)
    smaller_train_data_x = smaller_train_data

    # print(len(bigger_train_data), len(smaller_train_data))

    categroy_other = train_data[label_list]
    categroy_blood_sugar = train_data['categroy_blood_sugar']

    concat_blood_suger, pred_test_probability = probability_pridect(
        categroy_other, categroy_blood_sugar, test_data)
    # print(concat_blood_suger) #5641

    # 提取出来test数据
    categroy_blood_sugar_test = pd.DataFrame(
        concat_blood_suger[len(train_data):].reset_index())
    # print(categroy_blood_sugar_test)
    categroy_blood_sugar_test.drop(['index'], axis=1, inplace=True)
    # print(categroy_blood_sugar_test)
    test_data = pd.concat([test_data, categroy_blood_sugar_test], axis=1)

    return bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data, train_data, pred_test_probability


def get_model_byname(model_name):
    if model_name == 'lasso':
        model = make_pipeline(
            RobustScaler(), Lasso(
                alpha=0.005, random_state=1))
    elif model_name == 'ElasticNet':
        model = make_pipeline(
            RobustScaler(),
            ElasticNet(
                alpha=0.005,
                l1_ratio=.9,
                random_state=3))
    elif model_name == 'gdbt':
        model = GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features='sqrt',
            min_samples_leaf=15,
            min_samples_split=10,
            loss='huber',
            random_state=5)
    elif model_name == 'randomforest':
        model = RandomForestRegressor(
            n_estimators=1000,
            criterion='mse',
            max_depth=5,
            max_features=30,
            min_samples_leaf=8,
            n_jobs=12,
            random_state=17)
    elif model_name == "lgbm":
        model = LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=30)
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=5,
                                 min_child_weight=2, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.8, silent=True,
                                 random_state=7, nthread=-1)
    else:
        model = model_name()
    return model


def model_fit(
        model,
        bigger_train_data_x,
        bigger_train_data_y,
        smaller_train_data_x,
        smaller_train_data_y,
        test_data):
    model_for_biggger = model
    model_for_smaller = model

    model_for_biggger.fit(bigger_train_data_x, bigger_train_data_y)
    model_for_smaller.fit(smaller_train_data_x, smaller_train_data_y)

    prediction_by_bigger_model = model_for_biggger.predict(test_data)
    prediction_by_smaller_model = model_for_smaller.predict(test_data)

    return model_for_biggger, model_for_smaller, prediction_by_bigger_model, prediction_by_smaller_model


threshold = 6.5
num_test = test_data.shape[0]
num_train = train_data.shape[0]

bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data, train_data, pred_test_probability = probability_columns(
    train_data, test_data, threshold)

# regressor_models = [LinearRegression,
#                     'lasso',
#                     'ElasticNet']
#
# pred_test_regressor = np.zeros((num_test, len(regressor_models)))
# regressor_bigger = np.zeros(
#     (bigger_train_data_x.shape[0],
#      len(regressor_models)))
# regressor_smaller = np.zeros(
#     (smaller_train_data_x.shape[0],
#      len(regressor_models)))
#
# for i, model_name in enumerate(regressor_models):
#     model = get_model_byname(model_name)
#     print(model.__class__)
#     regressor_bigger[:, i], regressor_smaller[:, i], prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
#         model, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
#     pred_test_regressor[:, i] = np.array([prediction_by_smaller_model[i] *
#                                           pred_test_probability[i][0] +
#                                           prediction_by_bigger_model[i] *
#                                           pred_test_probability[i][1] for i in range(num_test)])

model_linear = get_model_byname(LinearRegression)
linear_bigger, linear_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_linear, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
linear_preds = np.array([prediction_by_smaller_model[i] *
                      pred_test_probability[i][0] +
                      prediction_by_bigger_model[i] *
                      pred_test_probability[i][1] for i in range(num_test)])

model_lasso = get_model_byname('lasso')
lasso_bigger, lasso_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_linear, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
lasso_preds = np.array([prediction_by_smaller_model[i] *
                      pred_test_probability[i][0] +
                      prediction_by_bigger_model[i] *
                      pred_test_probability[i][1] for i in range(num_test)])

model_ElasticNet = get_model_byname('ElasticNet')
ElasticNet_bigger, ElasticNet_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_linear, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
ElasticNet_preds = np.array([prediction_by_smaller_model[i] *
                      pred_test_probability[i][0] +
                      prediction_by_bigger_model[i] *
                      pred_test_probability[i][1] for i in range(num_test)])

model_rdf = get_model_byname('randomforest')
rdf_bigger, rdf_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_rdf, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
rdf_preds = np.array([prediction_by_smaller_model[i] *
                      pred_test_probability[i][0] +
                      prediction_by_bigger_model[i] *
                      pred_test_probability[i][1] for i in range(num_test)])

model_gdbt = get_model_byname('gdbt')
gdbt_bigger, gdbt_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_gdbt, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
gdbt_preds = np.array([prediction_by_smaller_model[i] *
                       pred_test_probability[i][0] +
                       prediction_by_bigger_model[i] *
                       pred_test_probability[i][1] for i in range(num_test)])

model_lgbm = get_model_byname("lgbm")
lgbm_bigger, lgbm_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_lgbm, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
lgbm_preds = np.array([prediction_by_smaller_model[i] *
                       pred_test_probability[i][0] +
                       prediction_by_bigger_model[i] *
                       pred_test_probability[i][1] for i in range(num_test)])


model_xgboost = get_model_byname('xgboost')
xgboost_bigger, xgboost_smaller, prediction_by_bigger_model, prediction_by_smaller_model = model_fit(
    model_lgbm, bigger_train_data_x, bigger_train_data_y, smaller_train_data_x, smaller_train_data_y, test_data)
xgboost_preds = np.array([prediction_by_smaller_model[i] *
                       pred_test_probability[i][0] +
                       prediction_by_bigger_model[i] *
                       pred_test_probability[i][1] for i in range(num_test)])


print("stacking ....")

# stacked_averaged_bigger_models = StackingRegressor(
#     regressors=[regressor_bigger[:, i] for i in range(len(regressor_models))],
#     meta_regressor= gdbt_bigger
# )
# stacked_averaged_less_models = StackingRegressor(
#     regressors=[regressor_smaller[:, i] for i in range(len(regressor_models))],
#     meta_regressor= gdbt_smaller
# )

stacked_averaged_bigger_models = StackingRegressor(
    regressors=[linear_bigger, lasso_bigger, ElasticNet_bigger],
    meta_regressor= gdbt_bigger
)
stacked_averaged_less_models = StackingRegressor(
    regressors=[linear_smaller, lasso_smaller, ElasticNet_smaller],
    meta_regressor= gdbt_smaller
)

# 拟合模型
stacked_averaged_bigger_models.fit(bigger_train_data_x, bigger_train_data_y)
stacked_averaged_less_models.fit(smaller_train_data_x, smaller_train_data_y)
# 测试集预测
stacked_bigger_pred = stacked_averaged_bigger_models.predict(test_data)
stacked_less_pred = stacked_averaged_less_models.predict(test_data)
# 预测结果结合权重
stacked_preds = np.array(
    [stacked_less_pred[i] * pred_test_probability[i][0] + stacked_bigger_pred[i] * pred_test_probability[i][1] for i in range(num_test)])

ensemble = stacked_preds * 0.40 + xgboost_preds * 0.40 + lgbm_preds * 0.20
print(
    '实际mse： {}'.format(
        metrics.mean_squared_error(
            df_test_A_answer,
            ensemble) *
        0.5))

# stacking融合linear
new_ensemble = np.array(
    [linear_preds[i] * pred_test_probability[i][0] + ensemble[i] * pred_test_probability[i][1] for i in range(num_test)])

print(
    '实际mse： {}'.format(
        metrics.mean_squared_error(
            df_test_A_answer,
            new_ensemble) *
        0.5))

# 实际mse： 1.5163544660602217
# 实际mse： 2.366871874472643