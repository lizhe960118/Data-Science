#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/2 12:04
@Author  : LI Zhe
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/29 16:30
@Author  : LI Zhe
"""
# 得到新的训练集的特征
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

data_train = pd.read_csv('../data/after_pre_test.csv', low_memory=False, encoding='gbk')

train_x = data_train.iloc[:,:-1]
train_y = data_train.iloc[:,-1]

# model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=2000, max_depth=9, min_child_weight=5, seed=0,
#                          subsample=0.8, colsample_bytree=0.8, gamma=0.2, reg_alpha=0, reg_lambda=1, metrics='mae')
# model.fit(train_x, train_y)
# new_X_train = model.apply(train_x)
# print(new_X_train.shape)
#
# new_train_feature = pd.Data_trainFrame(new_X_train)
# new_train_feature.to_csv('../data_train/new_train_feature.csv', index=False)

# def createFeaturebydivide(Feature, new_Feature):
#     tmp_Feature = []
#     for row in range(0, len(data_train[Feature])):
#         tmp = np.array([ data_train[Feature][row]/ data_train[new_Feature][row] ])
#         tmp_Feature.append(tmp)
#     return tmp_Feature

data_train['总胆固醇/高密度脂蛋白胆固醇'] = data_train['总胆固醇']/data_train['高密度脂蛋白胆固醇']
data_train['尿素/肌酐'] = data_train['尿素']/data_train['肌酐']
data_train['肾'] = data_train['尿素'] + data_train['肌酐'] + data_train['尿酸']
data_train['红细胞计数*红细胞平均血红蛋白量'] = data_train['红细胞计数'] * data_train['红细胞平均血红蛋白量']
data_train['红细胞计数*红细胞平均血红蛋白浓度'] = data_train['红细胞计数'] * data_train['红细胞平均血红蛋白浓度']
data_train['红细胞计数*红细胞平均体积'] = data_train['红细胞计数'] * data_train['红细胞平均体积']
data_train['嗜酸细胞'] = data_train['嗜酸细胞%'] * 100
data_train['年龄*中性粒细胞%/尿酸*血小板比积'] = data_train['年龄'] * \
    data_train['中性粒细胞%'] / (data_train['尿酸'] * data_train['血小板比积'])
# print(data_train[['总胆固醇/高密度脂蛋白胆固醇']].info(), data_train[['尿素/肌酐']].info())

# new_train_feature = pd.concat([data_train['总胆固醇/高密度脂蛋白胆固醇'],data_train['尿素/肌酐']])
# new_train_feature = pd.concat([data_train, new_train_feature])

# xuetang_temp = data_train['血糖']
# data_train = data_train.drop(["血糖"], axis=1)
# data_train['血糖'] = xuetang_temp

data_train.to_csv('../data/after_pre_test_20180930.csv', index=False)