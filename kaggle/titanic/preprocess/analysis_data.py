from sklearn import preprocessing
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
sns.set_style('whitegrid')
# train_data.head()
# train_data.info()
print("-" * 40)
# test_data.info()

# 总体的存活率
# train_data['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
# plt.show()

# 分析数据关系
# # 性别和存活率的关系
# train_data.groupby(['Sex', 'Survived'])['Survived'].count()
# train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
# plt.show()

# # 存活与船舱等级之间的关系
# train_data.groupby(['Pclass', 'Survived'])['Pclass'].count()
# train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
# plt.show()

# # 存活与船舱等级、性别之间的关系
# train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count()
# train_data[['Sex', 'Pclass', 'Survived']].groupby(
#     ['Pclass', 'Sex']).mean().plot.bar()
# plt.show()

# 分析不同等级船舱下, 年龄分布和生存的关系
# 分析不同性别下, 年龄分布和生存的关系‘
# fig, ax = plt.subplots(1, 2, figsize=(18, 8))
# sns.violinplot(
#     "Pclass",
#     "Age",
#     hue="Survived",
#     data=train_data,
#     split=True,
#     ax=ax[0])
# ax[0].set_title('Pclass and Age vs Survived')
# ax[0].set_yticks(range(0, 110, 10))
# sns.violinplot(
#     "Sex",
#     "Age",
#     hue="Survived",
#     data=train_data,
#     split=True,
#     ax=ax[1])
# ax[1].set_title('Sex and Age vs Survived')
# ax[1].set_yticks(range(0, 110, 10))
# plt.show()


# # 分析总体的年龄分布
# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# train_data['Age'].hist(bins=70)
# plt.xlabel('Age')
# plt.ylabel('Num')
# plt.subplot(122)
# train_data.boxplot(column='Age', showfliers=False)
# plt.show()


# 不同年龄下的平均生存率
train_data_df = pd.DataFrame(train_data)
train_data_age_not_null = train_data_df[train_data_df['Age'].notnull()]

# # 不同年龄下的生存和非生存的分布情况
# facet = sns.FacetGrid(train_data_age_not_null, hue="Survived", aspect=4)
# facet.map(sns.kdeplot, 'Age', shade=True)
# facet.set(xlim=(0, train_data['Age'].max()))
# facet.add_legend()
# plt.show()

# 不同年龄下的存活率
# fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
# train_data_age_not_null["Age_int"] = train_data_age_not_null["Age"].astype(int)
# average_age = train_data_age_not_null[["Age_int", "Survived"]].groupby(
#     ['Age_int'], as_index=False).mean()
# sns.barplot(x='Age_int', y='Survived', data=average_age)
# print(train_data_age_not_null['Age'].describe())
# plt.show()

# # 按照年龄，将乘客划分为儿童、少年、成年和老年，分析四个群体的生还情况
# # 不同年龄群体的存活率
# bins = [0, 12, 18, 65, 100]
# train_data_age_not_null['Age_group'] = pd.cut(train_data_age_not_null['Age'], bins)
# by_age = train_data_age_not_null.groupby('Age_group')['Survived'].mean()
# by_age.plot(kind='bar')
# plt.show()

# # 称呼与存活与否的关系 Name
# train_data['Title'] = train_data['Name'].str.extract(
#     ' ([A-Za-z]+)\.', expand=False)
# pd.crosstab(train_data['Title'], train_data['Sex'])
# train_data[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
# plt.show()

# # 名字长度与存活之间的关系
# # 不同名字长度下的存活率
# fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
# train_data['Name_length'] = train_data['Name'].apply(len)
# name_length = train_data[['Name_length', 'Survived']].groupby(
#     ['Name_length'], as_index=False).mean()
# sns.barplot(x='Name_length', y='Survived', data=name_length)
# plt.show()

# # 有无兄弟姐妹和存活与否的关系 SibSp
# # 将数据分为有兄弟姐妹的和没有兄弟姐妹的两组：
# sibsp_df = train_data[train_data['SibSp'] != 0]
# no_sibsp_df = train_data[train_data['SibSp'] == 0]
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# sibsp_df['Survived'].value_counts().plot.pie(
#     labels=['No Survived', 'Survived'], autopct='%1.1f%%')
# plt.xlabel('sibsp')
# plt.subplot(122)
# no_sibsp_df['Survived'].value_counts().plot.pie(
#     labels=['No Survived', 'Survived'], autopct='%1.1f%%')
# plt.xlabel('no_sibsp')
# plt.show()

# # 有无父母子女和存活与否的关系 Parch
# parch_df = train_data[train_data['Parch'] != 0]
# no_parch_df = train_data[train_data['Parch'] == 0]
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# parch_df['Survived'].value_counts().plot.pie(
#     labels=['No Survived', 'Survived'], autopct='%1.1f%%')
# plt.xlabel('parch')
# plt.subplot(122)
# no_parch_df['Survived'].value_counts().plot.pie(
#     labels=['No Survived', 'Survived'], autopct='%1.1f%%')
# plt.xlabel('no_parch')
# plt.show()

# # 亲友的人数和存活与否的关系 SibSp、Parch
# fig, ax = plt.subplots(1, 2, figsize=(18, 8))
# train_data[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
# ax[0].set_title('Parch and Survived')
# train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
# ax[1].set_title('SibSp and Survived')
# plt.show()
# # 亲友的人数和存活与否的关系 SibSp & Parch
# train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
# train_data[['Family_Size', 'Survived']].groupby(
#     ['Family_Size']).mean().plot.bar()
# plt.show()

# # 票价分布
# plt.figure(figsize=(10, 5))
# train_data['Fare'].hist(bins=70)
# train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
# plt.show()
# print(train_data['Fare'].describe())

# # 平均票价分布和存活与否的关系 Fare
# fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
# fare_survived = train_data['Fare'][train_data['Survived'] == 1]
# average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
# std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
# average_fare.plot(yerr=std_fare, kind='bar', legend=False)
# plt.xlabel('Survived')
# plt.show()

# 船舱类型为U0和存活的关系
train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
# train_data['Has_Cabin'] = train_data['Cabin'].apply(
#     lambda x: 0 if x == 'U0' else 1)
# train_data[['Has_Cabin', 'Survived']].groupby(['Has_Cabin']).mean().plot.bar()
# plt.show()

#  船舱类型和存活与否的关系 Cabin
# Replace missing values with "U0"
# # create feature for the alphabetical part of the cabin number
# train_data['CabinLetter'] = train_data['Cabin'].map(
#     lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# # convert the distinct cabin letters with incremental integer values
# train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
# train_data[['CabinLetter', 'Survived']].groupby(
#     ['CabinLetter']).mean().plot.bar()
# plt.show()

# # 港口和存活数量的关系
# sns.countplot('Embarked', hue='Survived', data=train_data)
# plt.title('Embarked and Survived')
# #  港口和存活与否的关系 Embarked
# sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
# plt.title('Embarked and Survived rate')
# plt.show()