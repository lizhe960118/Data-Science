#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/27 22:12
@Author  : LI Zhe
"""
from catboost import CatBoostRegressor
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import metrics

train_df = pd.read_csv('../data/after_pre_train_20180928.csv', encoding='gb2312')
train_xy, val = train_test_split(train_df, test_size=0.3, random_state=1)

# 训练集
train_y = train_xy['血糖']
train_x = train_xy.drop(['血糖'], axis=1)
# 验证集
val_y = val['血糖']
val_x = val.drop(['血糖'], axis=1)

test = pd.read_csv('../data/after_pre_test.csv', encoding='gb2312')

cat_model = CatBoostRegressor(iterations= 20 * 40, learning_rate=0.03, depth=6, l2_leaf_reg=1,  random_seed= 0)
cat_model.fit(train_x, train_y)
# preds = cat_model.predict(val_x)
# print(metrics.mean_squared_error(val_y, preds) / 2)  # 0.8387019297676944

df_test = pd.read_csv(
    '../data/after_pre_test.csv',
    encoding='gbk')
test_data = pd.DataFrame(df_test)

df_test_A_answer = pd.read_csv(
    '../data/d_answer_a_20180128.csv', header=-1)

prediction = cat_model.predict(test_data)
print(
    '实际mse： {}'.format(
        metrics.mean_squared_error(
            df_test_A_answer,
            prediction) *
        0.5))