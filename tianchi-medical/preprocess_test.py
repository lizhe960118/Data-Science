#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/26 18:08
@Author  : LI Zhe
"""
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestRegressor

data_origin = pd.read_csv(
    'data/d_test_A_20180102.csv',
    low_memory=False,
    encoding='gbk')

# 删除无用的特征
data_origin = data_origin.drop(
    ['id', '体检日期', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)

# 更改为Onehot特征
# from sklearn.preprocessing import OneHotEncoder
# #哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
# OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))

data_origin['性别'] = data_origin['性别'].apply(lambda x: 1 if x == '男' else 0)
# dummies_sex = pd.get_dummies(data_origin['性别'], prefix='sex')
# data_origin = pd.concat([data_origin, dummies_sex], axis=1, join_axes=[data_origin.index])
# data_origin = data_origin.drop(['性别'], axis=1)

# 填充缺失值
print(data_origin.isnull().sum())

"""
性别               0
年龄               0
*天门冬氨酸氨基转换酶    185
*丙氨酸氨基转换酶      185
*碱性磷酸酶         185
*r-谷氨酰基转换酶     185
*总蛋白           185
白蛋白            185
*球蛋白           185
白球比例           185
甘油三酯           176
总胆固醇           176
高密度脂蛋白胆固醇      176
低密度脂蛋白胆固醇      176
尿素             194
肌酐             194
尿酸             194
白细胞计数            5
红细胞计数            5
血红蛋白             5
红细胞压积            5
红细胞平均体积          5
红细胞平均血红蛋白量       5
红细胞平均血红蛋白浓度      5
红细胞体积分布宽度        5
血小板计数            5
血小板平均体积          6
血小板体积分布宽度        6
血小板比积            6
中性粒细胞%           5
淋巴细胞%            5
单核细胞%            5
嗜酸细胞%            5
嗜碱细胞%            5
dtype: int64
"""
"""
# 不太重要的特征，使用平均值填充
红细胞计数          151
血红蛋白           129
红细胞压积          109
中性粒细胞%         100
淋巴细胞%           94
红细胞平均血红蛋白量      93
血小板体积分布宽度       90
单核细胞%           88
血小板平均体积         88
血小板计数           79
血小板比积           63
嗜碱细胞%           57
嗜酸细胞%           53
"""


def fill_unimportant_feature(feature_name):
    featureName = data_origin[data_origin[feature_name].notnull()]
    featureName_mean = featureName[feature_name].mean()
    data_origin.loc[(data_origin[feature_name].isnull()),
                    feature_name] = featureName_mean


not_important_features = [
    '红细胞计数',
    '血红蛋白',
    '红细胞压积',
    '中性粒细胞%',
    '淋巴细胞%',
    '红细胞平均血红蛋白量',
    '血小板体积分布宽度',
    '单核细胞%',
    '血小板平均体积',
    '血小板计数',
    '血小板比积',
    '嗜碱细胞%',
    '嗜酸细胞%']

for feature_name in not_important_features:
    fill_unimportant_feature(feature_name)

"""
比较重要的特征，和缺失值较多的特征， 使用随机森林回归预测值填充
# 年龄                    378      不用填充
# 尿酸                    352      1378
# 甘油三酯                332      1219
# *天门冬氨酸氨基转换酶   307      1221
# 尿素                    283	   1378
# *碱性磷酸酶          	  245  	   1221
# 红细胞平均体积          237
# 红细胞平均血红蛋白浓度  206
# *r-谷氨酰基转换酶       204      1221
# 红细胞体积分布宽度      203
# *丙氨酸氨基转换酶       200      1221
# 白细胞计数              174
# 高密度脂蛋白胆固醇      174      1219
# 总胆固醇                143      1219
# 白蛋白                  140      1221
# 低密度脂蛋白胆固醇      128      1219
# *球蛋白                 118      1221
# 肌酐                    111      1378
# *总蛋白                 109      1221
# 白球比例                97       1221
"""

feature_stack = ['红细胞计数',
                 '血红蛋白',
                 '红细胞压积',
                 '中性粒细胞%',
                 '淋巴细胞%',
                 '红细胞平均血红蛋白量',
                 '血小板体积分布宽度',
                 '单核细胞%',
                 '血小板平均体积',
                 '血小板计数',
                 '血小板比积',
                 '嗜碱细胞%',
                 '嗜酸细胞%']

improtant_features = ['尿酸',
                      '甘油三酯',
                      '*天门冬氨酸氨基转换酶',
                      '尿素',
                      '*碱性磷酸酶',
                      '红细胞平均体积',
                      '红细胞平均血红蛋白浓度',
                      '*r-谷氨酰基转换酶',
                      '红细胞体积分布宽度',
                      '*丙氨酸氨基转换酶',
                      '白细胞计数',
                      '高密度脂蛋白胆固醇',
                      '总胆固醇',
                      '白蛋白',
                      '低密度脂蛋白胆固醇',
                      '*球蛋白',
                      '肌酐',
                      '*总蛋白',
                      '白球比例']


def fill_improtant_features(feature_stack, feature_name):
    df_improtant = data_origin[feature_stack]
    null_feature = df_improtant[df_improtant[feature_name].isnull()]
    notnull_feature = df_improtant[df_improtant[feature_name].notnull()]
    null_matrix = null_feature.as_matrix()
    notnull_matrix = notnull_feature.as_matrix()
    y = notnull_matrix[:, 0]
    X = notnull_matrix[:, 1:]
    null_use = null_matrix[:, 1:]
    clf = RandomForestRegressor(oob_score=True, random_state=10)
    clf.fit(X, y)
    pred = clf.predict(null_use)
    return pred


for feature_name in reversed(improtant_features):
    feature_stack.insert(0, feature_name)
    data_origin.loc[(data_origin[feature_name].isnull()),
                    feature_name] = fill_improtant_features(feature_stack,
                                                            feature_name)

data_origin.to_csv("data/after_pre_test.csv", index=False, header=True)

