import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.grid_search import GridSearchCV  # Perforing grid search
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# 载入数据
train_data = pd.read_csv(
    'data/after_pre_train_20180929.csv',
    low_memory=False,
    encoding='gbk')

# train_xy, val = train_test_split(train_data, test_size=0.3, random_state=1)

# 如果直接训练
y_train = train_data['血糖']
train_data.drop(['血糖'], axis=1, inplace=True)
y_train = y_train.as_matrix()
x_train = train_data.as_matrix()

# # 一折交叉验证
# # 训练集
# train_y = train_xy['血糖']
# train_x = train_xy.drop(['血糖'], axis=1)
# # 验证集
# val_y = val['血糖']
# val_x = val.drop(['血糖'], axis=1)


# 读取测试数据
test_data = pd.read_csv(
    'data/after_pre_test.csv',
    low_memory=False,
    encoding='gbk')

seed = 0
kfold = model_selection.KFold(n_splits=5, random_state=seed)


def get_model_byname(model_name):
    if model_name == 'KNeighborsRegressor':
        model = KNeighborsRegressor(n_neighbors=20)
    elif model_name == 'Lasso':
        model = Lasso(max_iter=10000, alpha=0.1)
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(
            learning_rate=0.1,
            n_estimators=84,
            max_depth=3,
            min_child_weight=6,
            seed=0,
            subsample=0.9,
            colsample_bytree=0.6,
            gamma=0.5,
            reg_alpha=2,
            reg_lambda=3)
    elif model_name == 'gdbt':
        params = {'n_estimators': 600, 'max_depth': 2, 'min_samples_split': 4,
                  'learning_rate': 0.025, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif model_name == 'randomforest':
        model = RandomForestRegressor(
            bootstrap=True,
            criterion='mse',
            max_depth=15,
            max_features=9,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=20,
            min_samples_split=10,
            min_weight_fraction_leaf=0.0,
            n_estimators=130,
            n_jobs=1,
            oob_score=False,
            random_state=0,
            verbose=0,
            warm_start=False)
    elif model_name == "catboost":
        model = CatBoostRegressor(iterations=20 * 40, learning_rate=0.03, depth=6, l2_leaf_reg=1, random_seed=0)
    else:
        model = model_name()
    return model


for model_name in [
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        KNeighborsRegressor,
        DecisionTreeRegressor,
        SVR,
        'xgboost',
        'gdbt',
        'randomforest',
        'catboost'
]:
    model = get_model_byname(model_name)
    results = model_selection.cross_val_score(
        model, x_train, y_train, cv=kfold, scoring = 'neg_mean_squared_error')
    print(model_name, abs(results.mean())/2)

# for model_name in [
#         'xgboost',
#         'gdbt',
#         'randomforest',
#         'catboost'
# ]:
    # model = get_model_byname(model_name)
    # sum_score = 0
    # for train_index, val_index in kfold.split(x_train):
    #     X_train, X_val = x_train[train_index], x_train[val_index]
    #     Y_train, Y_val = y_train[train_index], y_train[val_index]
    #     model.fit(X_train, Y_train)
    #     pre = model.predict(X_val)
    #     mse = metrics.mean_squared_error(Y_val, pre)
    #     sum_score += mse
    #     print("once")
    # print("avg_mse=", sum_score / 5)
 #   	pred = model.predict(x_test)
    # np.savetxt('../data/predict_%s.csv' % model_name, pred, fmt='%f')

# <class 'sklearn.linear_model.base.LinearRegression'> 2.011333174499831
# <class 'sklearn.linear_model.ridge.Ridge'> 2.007154715469367
# <class 'sklearn.linear_model.coordinate_descent.Lasso'> 2.1528424053155932
# <class 'sklearn.linear_model.coordinate_descent.ElasticNet'> 2.1259007449506084
# <class 'sklearn.neighbors.regression.KNeighborsRegressor'> 2.4336065997248553
# <class 'sklearn.tree.tree.DecisionTreeRegressor'> 4.169136128799728
# <class 'sklearn.svm.classes.SVR'> 2.4474301032829855
# xgboost 2.0245972347605257
# randomforest 2.043141174861236
# catboost 1.9957702108759103

# 20180929
# <class 'sklearn.linear_model.base.LinearRegression'> 1.8293311406729558
# <class 'sklearn.linear_model.ridge.Ridge'> 1.8257272246049023
# <class 'sklearn.linear_model.coordinate_descent.Lasso'> 1.9698362934275226
# <class 'sklearn.linear_model.coordinate_descent.ElasticNet'> 1.9474662726303578
# <class 'sklearn.neighbors.regression.KNeighborsRegressor'> 2.2111925188737915
# <class 'sklearn.tree.tree.DecisionTreeRegressor'> 4.12448606097155
# <class 'sklearn.svm.classes.SVR'> 2.254469114830362
# xgboost 1.8322628976178894
# gdbt 1.86504922531614
# randomforest 1.860631897504677
# catboost 1.8084265745423764

# 20180930
# <class 'sklearn.linear_model.base.LinearRegression'> 1.8359381620651454
# <class 'sklearn.linear_model.ridge.Ridge'> 1.8328775702049882
# <class 'sklearn.linear_model.coordinate_descent.Lasso'> 1.9420341241822456
# <class 'sklearn.linear_model.coordinate_descent.ElasticNet'> 1.9264480868086455
# <class 'sklearn.neighbors.regression.KNeighborsRegressor'> 2.34488597152944
# <class 'sklearn.tree.tree.DecisionTreeRegressor'> 3.8218884596611575