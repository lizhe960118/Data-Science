#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/29 12:49
@Author  : LI Zhe
"""
from pylab import mpl
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')

def plot_feature_imp(feature_name):
	mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
	train[feature_name] = train[feature_name].dropna()
	fig, ax = plt.subplots()
	ax.scatter(x=train[feature_name], y=train['血糖'])
	plt.ylabel('血糖')
	plt.xlabel(feature_name)
	plt.show()

# for feature_neme in [ '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']:
# for feature_name in ['总胆固醇','高密度脂蛋白胆固醇','尿素','肌酐']:
for feature_name in ['*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶','*碱性磷酸酶','*r-谷氨酰基转换酶']:
	plot_feature_imp(feature_name)

