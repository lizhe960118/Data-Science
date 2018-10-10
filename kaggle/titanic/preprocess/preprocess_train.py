from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('../data/train.csv')
train_data = pd.DataFrame(train_data)
# test_data = pd.read_csv('data/test.csv')
# test_data = pd.DataFrame(test_data)
sns.set_style('whitegrid')
# print(train_data.head())
# print(train_data.info())
# print("-" * 40)
# print(test_data.info())

print(train_data.isnull().sum())
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# print(test_data.isnull().sum())
# PassengerId      0
# Pclass           0
# Name             0
# Sex              0
# Age             86
# SibSp            0
# Parch            0
# Ticket           0
# Fare             1
# Cabin          327
# Embarked         0

# # 处理缺失值 Embarked
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values

# # 处理缺失值 Cabin 使用U0填充
# # replace missing value with U0
train_data['Cabin'] = train_data.Cabin.fillna('U0')
train_data.Cabin[train_data.Cabin.isnull()] = 'U0'

# 处理缺失值 age 使用随机森林预测
# choose training data to predict age
age_df = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:, 1:]
Y = age_df_notnull.values[:, 0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X, Y)
predictAges = RFR.predict(age_df_isnull.values[:, 1:])
train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges
print(train_data.info())

# 变量转换
# 定性(Qualitative)转换： Dummy Variables
embark_dummies = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)
# train_data.drop(['Embarked'], axis=1, inplace=True)
embark_dummies = train_data[['S', 'C', 'Q']]
print(embark_dummies.head())

# Factorizing
# create feature for the alphabetical part of the cabin number
train_data['CabinLetter'] = train_data['Cabin'].map(
    lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# convert the distinct cabin letters with incremental integer values
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
print(train_data['CabinLetter'].head())


# 定量(Quantitative)转换： Scaling
assert np.size(train_data['Age']) == 891
# StandardScaler will subtract the mean from each value then scale to the
# unit variance
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(
    train_data['Age'].values.reshape(-1, 1))
print(train_data['Age_scaled'].head())

# Divide all fares into quartiles
train_data['Fare'] = train_data[['Fare']].fillna(
    train_data.groupby('Pclass').transform(np.mean))
train_data['Group_Ticket'] = train_data['Fare'].groupby(
    by=train_data['Ticket']).transform('count')
train_data['Fare'] = train_data['Fare'] / train_data['Group_Ticket']
train_data.drop(['Group_Ticket'], axis=1, inplace=True)
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
print(train_data['Fare_bin'].head())
train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(
    train_data['Fare_bin_id']).rename(
        columns=lambda x: 'Fare_' + str(x))
train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)
train_data.drop(['Fare_bin'], axis=1, inplace=True)

# 为了后面的特征分析，这里我们也将 Sex 特征进行factorizing
train_data['Sex'] = pd.factorize(train_data['Sex'])[0]
sex_dummies_df = pd.get_dummies(
    train_data['Sex'], prefix=train_data[['Sex']].columns[0])
train_data = pd.concat([train_data, sex_dummies_df], axis=1)

# 对Title进行factorizing
train_data['Title'] = train_data['Name'].map(
    lambda x: re.compile(", (.*?)\.").findall(x)[0])
title_Dict = {}
title_Dict.update(dict.fromkeys(
    ['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(
    ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
train_data['Title'] = train_data['Title'].map(title_Dict)
# 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing
train_data['Title'] = pd.factorize(train_data['Title'])[0]
title_dummies_df = pd.get_dummies(
    train_data['Title'], prefix=train_data[['Title']].columns[0])
train_data = pd.concat(
    [train_data, title_dummies_df], axis=1)

# 添加一些新的特征
train_data['Name_length'] = train_data['Name'].apply(len)

# Pclass
# 建立PClass Fare Category
def pclass_fare_category(
        df,
        pclass1_mean_fare,
        pclass2_mean_fare,
        pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

Pclass1_mean_fare = train_data['Fare'].groupby(
    by=train_data['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = train_data['Fare'].groupby(
    by=train_data['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = train_data['Fare'].groupby(
    by=train_data['Pclass']).mean().get([3]).values[0]

# 建立Pclass_Fare Category
train_data['Pclass_Fare_Category'] = train_data.apply(
    pclass_fare_category, args=(
        Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
pclass_level = preprocessing.LabelEncoder()  # 给每一项添加标签
pclass_level.fit(np.array(['Pclass1_Low',
                           'Pclass1_High',
                           'Pclass2_Low',
                           'Pclass2_High',
                           'Pclass3_Low',
                           'Pclass3_High']))
# 转换成数值
train_data['Pclass_Fare_Category'] = pclass_level.transform(
    train_data['Pclass_Fare_Category'])
# dummy 转换
pclass_dummies_df = pd.get_dummies(
    train_data['Pclass_Fare_Category']).rename(
        columns=lambda x: 'Pclass_' + str(x))
train_data = pd.concat(
    [train_data, pclass_dummies_df], axis=1)
# 将 Pclass 特征factorize化：
train_data['Pclass'] = pd.factorize(train_data['Pclass'])[0]

# Parch and SibSp
# 由前面的分析，我们可以知道，亲友的数量没有或者太多会影响到Survived。所以将二者合并为FamliySize这一组合项，同时也保留这两项。

def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'


train_data['Family_Size'] = train_data['Parch'] + \
    train_data['SibSp'] + 1
train_data['Family_Size_Category'] = train_data['Family_Size'].map(
    family_size_category)
le_family = preprocessing.LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
train_data['Family_Size_Category'] = le_family.transform(
    train_data['Family_Size_Category'])
family_size_dummies_df = pd.get_dummies(
    train_data['Family_Size_Category'], prefix=train_data[['Family_Size_Category']].columns[0])
train_data = pd.concat(
    [train_data, family_size_dummies_df], axis=1)

# # Age
# missing_age_df = pd.DataFrame(train_data[['Age',
#                                                    'Embarked',
#                                                    'Sex',
#                                                    'Title',
#                                                    'Name_length',
#                                                    'Family_Size',
#                                                    'Family_Size_Category',
#                                                    'Fare',
#                                                    'Fare_bin_id',
#                                                    'Pclass']])
# missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
# missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
# missing_age_test.head()
#
# train_data.loc[(train_data.Age.isnull(
# )), 'Age'] = fill_missing_in_age(missing_age_train, missing_age_test)

# Ticket
train_data['Ticket_Letter'] = train_data['Ticket'].str.split(
).str[0]
train_data['Ticket_Letter'] = train_data['Ticket_Letter'].apply(
    lambda x: 'U0' if x.isnumeric() else x)
# 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
train_data['Ticket_Number'] = train_data['Ticket'].apply(
    lambda x: pd.to_numeric(x, errors='coerce'))
train_data['Ticket_Number'].fillna(0, inplace=True)
# 将 Ticket_Letter factorize
train_data['Ticket_Letter'] = pd.factorize(
    train_data['Ticket_Letter'])[0]

# Cabin
train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
train_data['Cabin'] = train_data['Cabin'].apply(
    lambda x: 0 if x == 'U0' else 1)

# 一些数据的正则化
scale_age_fare = preprocessing.StandardScaler().fit(
    train_data[['Age', 'Fare', 'Name_length']])
train_data[['Age', 'Fare', 'Name_length']] = scale_age_fare.transform(
    train_data[['Age', 'Fare', 'Name_length']])
# 弃掉无用特征
# combined_data_backup = train_data
# train_data.drop([ 'PassengerId',
#                           'Embarked',
#                           'Sex',
#                           'Name',
#                           'Title',
#                           'Fare_bin_id',
#                           'Pclass_Fare_Category',
#                           'Parch',
#                           'SibSp',
#                           'Family_Size_Category',
#                           'Ticket'],
#                          axis=1,
#                          inplace=True)

# train_data.to_csv('data/after_pre_train_no_drop_20181002.csv', index=False)
# y_train = train_data['Survived']