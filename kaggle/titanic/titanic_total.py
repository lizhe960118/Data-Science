import numpy as np
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# # # --------------------------first part ---------------------------------#

"""
#主要是数据的可视化
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
# sns.set_style('whitegrid')
# train_data.head()
# train_data.info()
# print("-" * 40)
# test_data.info()

# train_data['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
# plt.show()

# 处理缺失值
train_data.Embarked[train_data.Embarked.isnull(
)] = train_data.Embarked.dropna().mode().values
# replace missing value with U0
train_data['Cabin'] = train_data.Cabin.fillna('U0')
train_data.Cabin[train_data.Cabin.isnull()] = 'U0'
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
# train_data.info()
"""

'''
fill_missing_message1 = pd.DataFrame(train_data[['PassengerId',
                                                 'Survived',
                                                 'Pclass',
                                                 'Name',
                                                 'Sex',
                                                 'Age',
                                                 'SibSp',
                                                 'Parch',
                                                 'Ticket',
                                                 'Fare',
                                                 'Cabin',
                                                 'Embarked'
                                                ]])
fill_missing_message1.to_csv('fill_missing_message1.csv', index=False, sep=',')
'''
# np.savetxt('fill_missing_message1.csv', fill_missing_message1, fmt='%f %f %f %f %f %f %f %f %f %f %f %f')


# train_data = pd.read_csv('fill_missing_message1.csv')

# # #分析数据关系

# # (1)存活与性别之间的关系
# train_data.groupby(['Sex', 'Survived'])['Survived'].count()
# train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
# plt.show()

# # 存活与船舱等级之间的关系
# train_data.groupby(['Pclass', 'Survived'])['Pclass'].count()
# train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
# train_data[['Sex', 'Pclass', 'Survived']].groupby(
#     ['Pclass', 'Sex']).mean().plot.bar()
# train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count()
# plt.show()

# # 分别分析不同等级船舱和不同性别下的年龄分布和生存的关系
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

# # (2)不同年龄下的生存和非生存的分布情况
# facet = sns.FacetGrid(train_data, hue="Survived", aspect=4)
# facet.map(sns.kdeplot, 'Age', shade=True)
# facet.set(xlim=(0, train_data['Age'].max()))
# facet.add_legend()
# plt.show()

# # 不同年龄下的平均生存率
# average survived passengers by age
# fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
# train_data["Age_int"] = train_data["Age"].astype(int)
# average_age = train_data[["Age_int", "Survived"]].groupby(
#     ['Age_int'], as_index=False).mean()
# sns.barplot(x='Age_int', y='Survived', data=average_age)
# train_data['Age'].describe()
# plt.show()

# # 按照年龄，将乘客划分为儿童、少年、成年和老年，分析四个群体的生还情况
# bins = [0, 12, 18, 65, 100]
# train_data['Age_group'] = pd.cut(train_data['Age'], bins)
# by_age = train_data.groupby('Age_group')['Survived'].mean()
# by_age.plot(kind='bar')
# plt.show()

# # (3)称呼与存活与否的关系 Name
# train_data['Title'] = train_data['Name'].str.extract(
#     ' ([A-Za-z]+)\.', expand=False)
# pd.crosstab(train_data['Title'], train_data['Sex'])
# train_data[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
# plt.show()

# # (4)名字长度与存活之间的关系
# fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
# train_data['Name_length'] = train_data['Name'].apply(len)
# name_length = train_data[['Name_length', 'Survived']].groupby(
#     ['Name_length'], as_index=False).mean()
# sns.barplot(x='Name_length', y='Survived', data=name_length)
# plt.show()

# # (5)有无兄弟姐妹和存活与否的关系 SibSp
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

# # (6)有无父母子女和存活与否的关系 Parch
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

# # (7)亲友的人数和存活与否的关系 SibSp & Parch
# fig, ax = plt.subplots(1, 2, figsize=(18, 8))
# train_data[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
# ax[0].set_title('Parch and Survived')
# train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
# ax[1].set_title('SibSp and Survived')
# train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
# train_data[['Family_Size', 'Survived']].groupby(
#     ['Family_Size']).mean().plot.bar()
# plt.show()

# # (8)票价分布和存活与否的关系 Fare
# plt.figure(figsize=(10, 5))
# train_data['Fare'].hist(bins=70)
# train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
# plt.show()
# train_data['Fare'].describe()
# fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
# fare_survived = train_data['Fare'][train_data['Survived'] == 1]
# average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
# std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
# average_fare.plot(yerr=std_fare, kind='bar', legend=False)
# plt.show()

# # (9) 船舱类型和存活与否的关系 Cabin
# # Replace missing values with "U0"
# train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
# train_data['Has_Cabin'] = train_data['Cabin'].apply(
#     lambda x: 0 if x == 'U0' else 1)
# train_data[['Has_Cabin', 'Survived']].groupby(['Has_Cabin']).mean().plot.bar()
# # create feature for the alphabetical part of the cabin number
# train_data['CabinLetter'] = train_data['Cabin'].map(
#     lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# # convert the distinct cabin letters with incremental integer values
# train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
# train_data[['CabinLetter', 'Survived']].groupby(
#     ['CabinLetter']).mean().plot.bar()
# plt.show()

# (10) 港口和存活与否的关系 Embarked
# sns.countplot('Embarked', hue='Survived', data=train_data)
# plt.title('Embarked and Survived')
# sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
# plt.title('Embarked and Survived rate')
# plt.show()
#
# # 4. 变量转换
# # 定性(Qualitative)转换： Dummy Variables
# # 定性(Quantitative)变量描述了物体的某一（不能被数学表示的）方面，Embarked就是一个例子
# # 定量(Qualitative)变量可以以某种方式排序，Age就是一个很好的列子

# embark_dummies = pd.get_dummies(train_data['Embarked'])
# train_data = train_data.join(embark_dummies)
# train_data.drop(['Embarked'], axis=1, inplace=True)
# embark_dummies = train_data[['S', 'C', 'Q']]
# print(embark_dummies.head())

# # Factorizing
# # dummy不好处理Cabin（船舱号）这种标称属性，因为他出现的变量比较多。
# # 所以Pandas有一个方法叫做factorize()，它可以创建一些数字，来表示类别变量，
# # 对每一个类别映射一个ID，这种映射最后只生成一个特征，不像dummy那样生成多个特征。
# # Replace missing values with "U0"
# train_data['Cabin'][train_data.Cabin.isnull()] = 'U0'
# # create feature for the alphabetical part of the cabin number
# train_data['CabinLetter'] = train_data['Cabin'].map(
#     lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# # convert the distinct cabin letters with incremental integer values
# train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
# print(train_data['CabinLetter'].head())

# # 定量(Quantitative)转换： Scaling
# # Scaling可以将一个很大范围的数值映射到一个很小的范围(通常是-1 - 1，或则是0 - 1)，
# # 很多情况下我们需要将数值做Scaling使其范围大小一样，否则大范围数值特征将会由更高的权重。
# # 比如：Age的范围可能只是0-100，而income的范围可能是0-10000000，在某些对数组大小敏感的模型中会影响其结果

# assert np.size(train_data['Age']) == 891
# # StandardScaler will subtract the mean from each value then scale to the
# # unit variance
# scaler = preprocessing.StandardScaler()
# train_data['Age_scaled'] = scaler.fit_transform(
#     train_data['Age'].values.reshape(-1, 1))
# print(train_data['Age_scaled'].head())

# # Binning
# # Binning通过观察“邻居”(即周围的值)将连续数据离散化。
# # 存储的值被分布到一些“桶”或“箱“”中，就像直方图的bin将数据划分成几块一样。下面的代码对Fare进行Binning。
# # Divide all fares into quartiles

# train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
# print(train_data['Fare_bin'].head())

# # qcut() creates a new variable that identifies the quartile range
# # but we can't use the string# so either factorize or create dummies from the result factorize
# # 在将数据Bining化后，要么将数据factorize化，要么dummies化。

# train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
# # dummies
# fare_bin_dummies_df = pd.get_dummies(
#     train_data['Fare_bin']).rename(
#         columns=lambda x: 'Fare_' + str(x))
#
# train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)

# # # --------------------------------first part end --------------------------------#

# # # --------------------------second part ---------------------------------#
train_df_org = pd.read_csv('data/train.csv')
test_df_org = pd.read_csv('data/test.csv')
test_df_org['Survived'] = 0
# combined_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']

"""
# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket

# # (1) Embarked 因为“Embarked”项的缺失值不多，所以这里我们以众数来填充：

combined_train_test['Embarked'].fillna(
    combined_train_test['Embarked'].mode().iloc[0], inplace=True)

# 为了后面的特征分析，这里我们将 Embarked 特征进行facrorizing

combined_train_test['Embarked'] = pd.factorize(
    combined_train_test['Embarked'])[0]

# 使用 pd.get_dummies 获取one-hot 编码

emb_dummies_df = pd.get_dummies(
    combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket,Embarked_0,Embarked_1,Embarked_2

# # (2)为了后面的特征分析，这里我们也将 Sex 特征进行facrorizing
combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
sex_dummies_df = pd.get_dummies(
    combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket,Embarked_0,Embarked_1,Embarked_2,Sex_0,Sex_1

# # (3)Name 首先先从名字中提取各种称呼 what is each person's title?
combined_train_test['Title'] = combined_train_test['Name'].map(
    lambda x: re.compile(", (.*?)\.").findall(x)[0])

# # 将各式称呼进行统一化处理：

title_Dict = {}
title_Dict.update(dict.fromkeys(
    ['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(
    ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)

# 使用dummy对不同的称呼进行分列
# 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing

combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
title_dummies_df = pd.get_dummies(
    combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat(
    [combined_train_test, title_dummies_df], axis=1)

# 增加名字长度的特征：

combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)

# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket,Embarked_0,Embarked_1,Embarked_2,Sex_0,Sex_1
# Title,Title_0,Title_1,Title_2,Title_3,Title_4,Title_5,Name_length

#(4) Fare
# 由前面分析可以知道，Fare项在测试数据中缺少一个值，所以需要对该值进行填充。
# 我们按照一二三等舱各自的均价来填充：
# 下面transform将函数np.mean应用到各个group中

combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
    combined_train_test.groupby('Pclass').transform(np.mean))

# 通过对Ticket数据的分析，我们可以看到部分票号数据有重复，同时结合亲属人数及名字的数据，和票价船舱等级对比，
# 我们可以知道购买的票中有家庭票和团体票，所以我们需要将团体票的票价分配到每个人的头上。

combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(
    by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare'] / \
    combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)

# 使用binning给票价分等级：

combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)

# 对于5个等级的票价我们也可以继续使用dummy为票价等级分列：

combined_train_test['Fare_bin_id'] = pd.factorize(
    combined_train_test['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(
    combined_train_test['Fare_bin_id']).rename(
        columns=lambda x: 'Fare_' + str(x))
combined_train_test = pd.concat(
    [combined_train_test, fare_bin_dummies_df], axis=1)
combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)

# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket,Embarked_0,Embarked_1,Embarked_2,Sex_0,Sex_1
# Title,Title_0,Title_1,Title_2,Title_3,Title_4,Title_5,Name_length,Fare_bin_id,Fare_0,Fare_1,Fare_2,Fare_3,Fare_4

# (5) Pclass
# Pclass这一项，其实已经可以不用继续处理了，我们只需要将其转换为dummy形式即可。
# 但是为了更好的分析问题，我们这里假设对于不同等级的船舱，各船舱内部的票价也说明了各等级舱的位置，
# 那么也就很有可能与逃生的顺序有关系。所以这里分出每等舱里的高价和低价位
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


Pclass1_mean_fare = combined_train_test['Fare'].groupby(
    by=combined_train_test['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combined_train_test['Fare'].groupby(
    by=combined_train_test['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combined_train_test['Fare'].groupby(
    by=combined_train_test['Pclass']).mean().get([3]).values[0]

# 建立Pclass_Fare Category

combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(
    pclass_fare_category, args=(
        Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)

pclass_level = LabelEncoder()  # 给每一项添加标签

pclass_level.fit(np.array(['Pclass1_Low',
                           'Pclass1_High',
                           'Pclass2_Low',
                           'Pclass2_High',
                           'Pclass3_Low',
                           'Pclass3_High']))

# 转换成数值

combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(
    combined_train_test['Pclass_Fare_Category'])

# dummy 转换

pclass_dummies_df = pd.get_dummies(
    combined_train_test['Pclass_Fare_Category']).rename(
        columns=lambda x: 'Pclass_' + str(x))
combined_train_test = pd.concat(
    [combined_train_test, pclass_dummies_df], axis=1)

# 将 Pclass 特征factorize化：
combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]

# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket,Embarked_0,Embarked_1,Embarked_2,Sex_0,Sex_1
# Title,Title_0,Title_1,Title_2,Title_3,Title_4,Title_5,Name_length,Fare_bin_id,Fare_0,Fare_1,Fare_2,Fare_3,Fare_4,
# Pclass_Fare_Category, Pclass_0,Pclass_1,Pclass_2,Pclass_3,Pclass_4,Pclass_5

# (6) Parch and SibSp
# 由前面的分析，我们可以知道，亲友的数量没有或者太多会影响到Survived。
# 所以将二者合并为FamliySize这一组合项，同时也保留这两项。


def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'


combined_train_test['Family_Size'] = combined_train_test['Parch'] + \
    combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(
    family_size_category)
le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(
    combined_train_test['Family_Size_Category'])
family_size_dummies_df = pd.get_dummies(
    combined_train_test['Family_Size_Category'], prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat(
    [combined_train_test, family_size_dummies_df], axis=1)

# print(combined_train_test.head())
# Age,Cabin,Embarked,Fare,Name,Parch,PassengerId,Pclass,Sex,SibSp,Survived,Ticket,Embarked_0,Embarked_1,Embarked_2,Sex_0,Sex_1
# Title,Title_0,Title_1,Title_2,Title_3,Title_4,Title_5,Name_length,Fare_bin_id,Fare_0,Fare_1,Fare_2,Fare_3,Fare_4,
# Pclass_Fare_Category, Pclass_0,Pclass_1,Pclass_2,Pclass_3,Pclass_4,Pclass_5,Family_Size,Family_Size_Category,
# Family_Size_Category_0,Family_Size_Category_1,Family_Size_Category_2

# Age,Cabin,Embarked,(Fare),(Name),Parch,PassengerId,Pclass,Sex,
# SibSp,Survived,Ticket,Title,Name_length,Fare_bin_id,Pclass_Fare_Category,Family_Size,Family_Size_Category

train_test_before_age = pd.DataFrame(combined_train_test[['PassengerId',
                                                          'Survived',
                                                          'Fare',
                                                          'Pclass_Fare_Category',
                                                          'Pclass',
                                                          'Title',
                                                          'Name_length',
                                                          'Sex',
                                                          'Age',
                                                          'Family_Size',
                                                          'Family_Size_Category',
                                                          'Parch',
                                                          'SibSp',
                                                          'Ticket',
                                                          'Fare_bin_id',
                                                          'Cabin',
                                                          'Embarked']
                                                         ])
train_test_before_age.to_csv('data/train_test_before_age.csv', index=False, sep=',')
"""

# combined_train_test = pd.read_csv('fill/train_test_before_age.csv')
# # (7)Age
# # 因为Age项的缺失值较多，所以不能直接填充age的众数或者平均数。
# # 常见的有两种对年龄的填充方式：
# # 一种是根据Title中的称呼，如Mr，Master、Miss等称呼不同类别的人的平均年龄来填充；
# # 一种是综合几项如Sex、Title、Pclass等其他没有缺失值的项，使用机器学习算法来预测Age。
# # 这里我们使用后者来处理。以Age为目标值，将Age完整的项作为训练集，将Age缺失的项作为测试集
#
# missing_age_df = pd.DataFrame(combined_train_test[['Age',
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
# # print(missing_age_test.head())
# missing_age_train.to_csv(
#     'data/missing_age_train.csv',
#     index=False,
#     sep=',')
# missing_age_test.to_csv(
#     'data/missing_age_test.csv',
#     index=False,
#     sep=',')

# missing_age_train = pd.read_csv('missing_age_train.csv')
# missing_age_test = pd.read_csv('missing_age_test.csv')

# combined_train_test = pd.read_csv('fill_age/train_test_after_age.csv')
# #
# # (8) Ticket
# # 观察Ticket的值，我们可以看到，Ticket有字母和数字之分，
# # 而对于不同的字母，可能在很大程度上就意味着船舱等级或者不同船舱的位置，也会对Survived产生一定的影响，
# # 所以我们将Ticket中的字母分开，为数字的部分则分为一类。
#
# combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split(
# ).str[0]
# combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(
#     lambda x: 'U0' if x.isnumeric() else x)
#
# # 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
# # combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(
# #   lambda x: pd.to_numeric(x, errors='coerce'))
# # combined_train_test['Ticket_Number'].fillna(0, inplace=True)
#
# # 将 Ticket_Letter factorize
#
# combined_train_test['Ticket_Letter'] = pd.factorize(
#     combined_train_test['Ticket_Letter'])[0]
#
#
# # (9)Cabin
# # 因为Cabin项的缺失值确实太多了，我们很难对其进行分析，或者预测。所以这里我们可以直接将Cabin这一项特征去除。
# # 但通过上面的分析，可以知道，该特征信息的有无也与生存率有一定的关系，
# # 所以这里我们暂时保留该特征，并将其分为有和无两类。
#
# combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
# combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(
#     lambda x: 0 if x == 'U0' else 1)
#
# print(combined_train_test.head())
#
# train_test_final = pd.DataFrame(combined_train_test[['PassengerId',
#                                                      'Survived',
#                                                      'Pclass_Fare_Category',
#                                                      'Pclass',
#                                                      'Title',
#                                                      'Name_length',
#                                                      'Sex',
#                                                      'Age',
#                                                      'Family_Size',
#                                                      'Family_Size_Category',
#                                                      'Parch',
#                                                      'SibSp',
#                                                      'Ticket_Letter',
#                                                      'Fare_bin_id',
#                                                      'Fare',
#                                                      'Cabin',
#                                                      'Embarked']
#                                                     ])
# train_test_final.to_csv('data/train_test_final.csv', index=False, sep=',')
#
#
# # 特征间相关性分析
# # 我们挑选一些主要的特征，生成特征之间的关联图，查看特征与特征之间的相关性：

# combined_train_test = pd.read_csv('data/train_test_final.csv')
# Correlation = pd.DataFrame(combined_train_test[['Embarked',
#                                                 'Sex',
#                                                 'Title',
#                                                 'Name_length',
#                                                 'Family_Size',
#                                                 'Family_Size_Category',
#                                                 'Fare',
#                                                 'Fare_bin_id',
#                                                 'Pclass',
#                                                 'Pclass_Fare_Category',
#                                                 'Age',
#                                                 'Ticket_Letter',
#                                                 'Cabin']])
#
# colormap = plt.cm.viridis
# plt.figure(figsize=(14, 12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(
#     Correlation.astype(float).corr(),
#     linewidths=0.1,
#     vmax=1.0,
#     square=True,
#     cmap=colormap,
#     linecolor='white',
#     annot=True)

# # 特征之间的数据分布图
# g = sns.pairplot(combined_train_test[[u'Survived',
#                                       u'Pclass',
#                                       u'Sex',
#                                       u'Age',
#                                       u'Fare',
#                                       u'Embarked',
#                                       u'Family_Size',
#                                       u'Title',
#                                       u'Ticket_Letter']],
#                  hue='Survived',
#                  palette='seismic',
#                  size=1.2,
#                  diag_kind='kde',
#                  diag_kws=dict(shade=True),
#                  plot_kws=dict(s=10))
# g.set(xticklabels=[])
# plt.show()


# # 一些数据的正则化
# # 这里我们将Age和fare进行正则化：
#
# X = combined_train_test[['Age', 'Fare', 'Name_length']]
# print(np.isnan(X).any())
# scale_age_fare = preprocessing.StandardScaler().fit(
#     combined_train_test[['Age', 'Fare', 'Name_length']].as_matrix().astype(np.float))
# combined_train_test[['Age', 'Fare', 'Name_length']] = scale_age_fare.transform(
#     combined_train_test[['Age', 'Fare', 'Name_length']].as_matrix().astype(np.float))
#
# # 弃掉无用特征
# # 对于上面的特征工程中，我们从一些原始的特征中提取出了很多要融合到模型中的特征，
# # 但是我们需要剔除那些原本的我们用不到的或者非数值特征。
# # 首先对我们的数据先进行一下备份，以便后期的再次分析
#
# # combined_data_backup = combined_train_test
# # combined_train_test.drop(['PassengerId',
# #                           'Embarked',
# #                           'Sex',
# #                           'Name',
# #                           'Title',
# #                           'Fare_bin_id',
# #                           'Pclass_Fare_Category',
# #                           'Parch',
# #                           'SibSp',
# #                           'Family_Size_Category',
# #                           'Ticket'],
# #                          axis=1,
# #                          inplace=True)
#
# 将训练数据和测试数据分开：
# train_data = combined_train_test[:891]
# test_data = combined_train_test[891:]
#
# titanic_train_data_X = train_data.drop(['Survived'], axis=1)
# titanic_train_data_X.to_csv(
#     'data/titanic_train_data_X.csv',
#     index=False,
#     sep=',')
#
# titanic_train_data_Y = pd.DataFrame(train_data[['Survived']])
# titanic_train_data_Y.to_csv(
#     'data/titanic_train_data_Y.csv',
#     index=False,
#     sep=',')
#
# titanic_test_data_X = test_data.drop(['Survived'], axis=1)
# titanic_test_data_X.to_csv(
#     'data/titanic_test_data_X.csv',
#     index=False,
#     sep=',')
#
# print(titanic_train_data_X.shape)

# # # -------------------PART THREE-----------------------------
titanic_train_data_X = pd.read_csv('data/titanic_train_data_X.csv')
titanic_train_data_Y = pd.read_csv('data/titanic_train_data_Y.csv')
titanic_test_data_X = pd.read_csv('data/titanic_test_data_X.csv')

# feature_to_pick = 30
# feature_top_n, feature_importance = get_top_n_features(
#     titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
# titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
# titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

# 用视图可视化不同算法筛选的特征排序
# rf_feature_imp = feature_importance[:10]
# Ada_feature_imp = feature_importance[32:32 + 10].reset_index(drop=True)
# # make importances relative to max importance
# rf_feature_importance = 100.0 * \
#     (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())
# Ada_feature_importance = 100.0 * \
#     (Ada_feature_imp['importance'] / Ada_feature_imp['importance'].max())
# # Get the indexes of all features over the importance threshold
# rf_important_idx = np.where(rf_feature_importance)[0]
# Ada_important_idx = np.where(Ada_feature_importance)[0]
# # Adapted from Gradient Boosting regression
# pos = np.arange(rf_important_idx.shape[0]) + .5
# plt.figure(1, figsize=(18, 8))
# plt.subplot(121)
# plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])
# plt.yticks(pos, rf_feature_imp['feature'][::-1])
# plt.xlabel('Relative Importance')
# plt.title('RandomForest Feature Importance')
# plt.subplot(122)
# plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])
# plt.yticks(pos, Ada_feature_imp['feature'][::-1])
# plt.xlabel('Relative Importance')
# plt.title('AdaBoost Feature Importance')
# plt.show()

# 模型融合（Model Ensemble）
# 这里我们使用了两层的模型融合，
# Level 1使用了：RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM，一共7个模型
# Level 2使用了XGBoost使用第一层预测的结果作为特征对最终的结果进行预测
# Some useful parameters which will come in handy later on
ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0  # for reproducibility
NFOLDS = 7  # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)


def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


rf = RandomForestClassifier(
    n_estimators=500,
    warm_start=True,
    max_features='sqrt',
    max_depth=6,
    min_samples_split=3,
    min_samples_leaf=2,
    n_jobs=-1,
    verbose=0)
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
et = ExtraTreesClassifier(
    n_estimators=500,
    n_jobs=-1,
    max_depth=8,
    min_samples_leaf=2,
    verbose=0)
gb = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.008,
    min_samples_split=3,
    min_samples_leaf=2,
    max_depth=5,
    verbose=0)
dt = DecisionTreeClassifier(max_depth=8)
knn = KNeighborsClassifier(n_neighbors=2)
svm = SVC(kernel='linear', C=0.025)
# 将pandas转换为arrays：
# Create Numpy arrays of train, test and target (Survived) dataframes to
# feed into our models
x_train = titanic_train_data_X.values
# Creates an array of the train data
x_test = titanic_test_data_X.values
# Creates an array of the test data
y_train = titanic_train_data_Y.values
# Create our OOF train and test predictions. These base results will be
# used as new features
rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test)
# Random Forest
ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test)
# AdaBoost
et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test)
# Extra Trees
gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test)
# Gradient Boost
dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test)
# Decision Tree
knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test)
# KNeighbors
svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test)
# Support Vector
print("Training is complete")

# 预测并生成提交文件
x_train = np.concatenate(
    (rf_oof_train,
     ada_oof_train,
     et_oof_train,
     gb_oof_train,
     dt_oof_train,
     knn_oof_train,
     svm_oof_train),
    axis=1)
x_test = np.concatenate(
    (rf_oof_test,
     ada_oof_test,
     et_oof_test,
     gb_oof_test,
     dt_oof_test,
     knn_oof_test,
     svm_oof_test),
    axis=1)

gbm = XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(
    x_train,
    y_train)
predictions = gbm.predict(x_test)

# test_df_org = pd.read_csv('data/test.csv')
# PassengerId = test_df_org['PassengerId']

StackingSubmission = pd.DataFrame(
    {'PassengerId': PassengerId, 'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')
#
# # 验证：学习曲线
#
#
# def plot_learning_curve(estimator,
#                         title,
#                         X,
#                         y,
#                         ylim=None,
#                         cv=None,
#                         n_jobs=1,
#                         train_sizes=np.linspace(.1,
#                                                 1.0,
#                                                 5),
#                         verbose=0):
#     """
#     Generate a simple plot of the test and traning learning curve.
#     Parameters----------
#     estimator : object type that implements the "fit" and "predict" methods
#     An object of that type which is cloned for each validation.
#     title : string
#     Title for the chart.
#     X : array-like, shape (n_samples, n_features)
#     Training vector, where n_samples is the number of samples and
#     n_features is the number of features.
#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#     Target relative to X for classification or regression;
#     None for unsupervised learning.
#     ylim : tuple, shape (ymin, ymax), optional
#     Defines minimum and maximum yvalues plotted.
#     cv : integer, cross-validation generator, optional
#     If an integer is passed, it is the number of folds (defaults to 3).
#     Specific cross-validation objects can be passed, see
#     sklearn.cross_validation module for the list of possible objects
#     n_jobs : integer, optional
#     Number of jobs to run in parallel (default 1).
#     """
#
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt
#
#
# X = x_train
# Y = y_train  # RandomForest
# rf_parameters = {
#     'n_jobs': -1,
#     'n_estimators': 500,
#     'warm_start': True,
#     'max_depth': 6,
#     'min_samples_leaf': 2,
#     'max_features': 'sqrt',
#     'verbose': 0}
# # AdaBoost
# ada_parameters = {'n_estimators': 500, 'learning_rate': 0.1}
# # ExtraTrees
# et_parameters = {
#     'n_jobs': -1,
#     'n_estimators': 500,
#     'max_depth': 8,
#     'min_samples_leaf': 2,
#     'verbose': 0}
# # GradientBoosting
# gb_parameters = {
#     'n_estimators': 500,
#     'max_depth': 5,
#     'min_samples_leaf': 2,
#     'verbose': 0}
# # DecisionTree
# dt_parameters = {'max_depth': 8}
# # KNeighbors
# knn_parameters = {'n_neighbors': 2}
# # SVM
# svm_parameters = {'kernel': 'linear', 'C': 0.025}
# # XGB
# gbm_parameters = {
#     'n_estimators': 2000,
#     'max_depth': 4,
#     'min_child_weight': 2,
#     'gamma': 0.9,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'objective': 'binary:logistic',
#     'nthread': -1,
#     'scale_pos_weight': 1}
#
# title = "Learning Curves"
# plot_learning_curve(
#     RandomForestClassifier(
#         **rf_parameters),
#     title,
#     X,
#     Y,
#     cv=None,
#     n_jobs=4,
#     train_sizes=[50,
#                  100,
#                  150,
#                  200,
#                  250,
#                  350,
#                  400,
#                  450,
#                  500])
# plt.show()
