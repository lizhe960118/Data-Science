#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/29 9:59
@Author  : LI Zhe
"""
import numpy as np
import pandas as pd
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

import matplotlib.pyplot as plt
from pylab import mpl
from matplotlib import font_manager

zhfont = font_manager.FontProperties(
    fname="C:/Users/李大哲/Desktop/my_project/Tianchi-medical/importance_plot/DroidSansFallback.ttf")
# 加载数据
# train = pd.read_csv('../data/new_train_feature.csv', encoding='gbk')
train = pd.read_csv('../data/d_train_20180102.csv', encoding='gbk')
# train = train[train['血糖'] <25 ]

# 查看血糖分布
# sns.distplot(train['血糖'], fit=norm)
# (mu, sigma) = norm.fit(train['血糖'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# # Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(
#     mu, sigma)], loc='best')
# plt.ylabel('Frequency')
# plt.title('血糖分布')
# fig = plt.figure()
# res = stats.probplot(train['血糖'], plot=plt)
# plt.show()

# # 使用box cox使血糖分布趋近均值，此时无太大意义
# tmp, lambda_  =  stats.boxcox(train['血糖'])
# print(len(tmp))
# sns.distplot(tmp, fit=norm)
# (mu, sigma) = norm.fit(tmp)
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist:($\mu=$ {:.2f} and $\sigma=$ {:.7f}'.format(mu, sigma)], loc='best')
# plt.ylabel("Frequence")
# fig = plt.figure()
# res = stats.probplot(tmp, plot=plt)
# plt.show()

# 查看特征之间的相关性
# corrmat = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,vmax=0.9,square=True)
# plt.show()

# corrmat = train.corr()
# f, ax = plt.subplots(figsize=(20, 9))
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontproperties=zhfont)#
# ax.set_yticklabels(ax.get_yticklabels(), rotation=180, fontproperties=zhfont)
# sns.heatmap(corrmat, vmax=0.9, yticklabels=True, square=True)
# locs, labels = plt.yticks()
# plt.setp(labels, rotation=360)
# plt.show()

# 查看关系大的图像
# corrmat = train.corr()
# k = 10  # number of variables for heatmap
# f, ax = plt.subplots(figsize=(20, 9))
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontproperties=zhfont)
# ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zhfont)
# cols = corrmat.nlargest(k, '血糖')['血糖'].index
# cm = np.corrcoef(train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
#                  'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# locs, labels = plt.yticks()
# plt.setp(labels, rotation=360)
# plt.show()

# 查看缺失率
total_null = train.isnull().sum().sort_values(ascending=False)
percent_null = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent_null],axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))