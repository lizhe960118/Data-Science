#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/2 15:04
@Author  : LI Zhe
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('data/after_pre_train_no_drop_20181002.csv')
test_data = pd.read_csv('data/after_pre_test_no_drop_20181002.csv')
test_data['Survived'] = 0

combined_train_test = train_data.append(test_data)
combined_train_test.to_csv("data/combined_train_test.csv", index=False)

# 特征间相关性分析
Correlation = pd.DataFrame(combined_train_test[['Sex',
                                                'Title',
                                                'Name_length',
                                                'Family_Size',
                                                'Family_Size_Category',
                                                'Fare',
                                                'Fare_bin_id',
                                                'Pclass',
                                                'Pclass_Fare_Category',
                                                'Age',
                                                'Ticket_Letter',
                                                'Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(
    Correlation.astype(float).corr(),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor='white',
    annot=True)
plt.show()

# 特征之间的数据分布图
g = sns.pairplot(combined_train_test[[u'Survived',
                                      u'Pclass',
                                      u'Sex',
                                      u'Age',
                                      u'Fare',
                                      # u'Embarked',
                                      u'Family_Size',
                                      u'Title',
                                      u'Ticket_Letter']],
                 hue='Survived',
                 palette='seismic',
                 size=1.2,
                 diag_kind='kde',
                 diag_kws=dict(shade=True),
                 plot_kws=dict(s=10))
g.set(xticklabels=[])
plt.show()