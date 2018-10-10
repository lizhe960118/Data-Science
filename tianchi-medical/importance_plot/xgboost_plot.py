#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/26 16:01
@Author  : LI Zhe
"""
# 查看特征的重要性
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

data_origin = pd.read_csv('../data/d_train_20180102.csv', low_memory=False, encoding='gbk')
data_origin = data_origin.drop(['id', '体检日期','乙肝表面抗原', '乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'], axis=1)
data_origin['性别'] = data_origin['性别'].apply(lambda x: 1 if x == '男' else 0)

# data_origin = pd.read_csv('../data/new_train_feature.csv', low_memory=False, encoding='gbk')

# base_param = {
# 	"booster":"gbtree",
# 	"max_depth":5,
# 	"eta":0.02,
# 	"seed":710,
# 	"objective":"reg:linear",
# 	"gamma":0.9,
# 	"min_child_weight":5,
# 	"subsample":0.8,
# 	"colsample_bytree":0.8
# }

base_param = {
	"eta" :0.05,
	'max_depth':8,
	"subsample":0.7,
	"colsample_bytree":0.7,
	"objective":"reg:linear",
	"silent":1,
	'seed':0
}


train_x = data_origin.iloc[:,:-1]
train_y = data_origin.iloc[:,-1]

dtrain = xgb.DMatrix(train_x.values, train_y.values, feature_names=train_x.columns)
watchlist = [(dtrain, 'train')]
xgb_model = xgb.train(base_param, dtrain, 256, watchlist, early_stopping_rounds=100)

feat_imp = pd.Series(xgb_model.get_fscore()).sort_values(ascending=False)
print(feat_imp)

feat_imp.plot(kind='bar', title='Feature importance')
# plt.plot(feat_imp, linewidth=5, kind='bar', title='Feature importance')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("Feature improtance score")
plt.show()