#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    """
     @Time    : 2018/9/27 22:00
    @Author  : LI Zhe
    """
    import numpy as np
    from sklearn import ensemble
    from sklearn.cross_validation import train_test_split
    import pandas as pd
    from sklearn import metrics
    from sklearn.grid_search import GridSearchCV  # Perforing grid search

    train_df = pd.read_csv('../data/after_pre_train_20180928.csv', encoding='gb2312')
    train_xy, val = train_test_split(train_df, test_size=0.3, random_state=1)

    # 训练集
    train_y = train_xy['血糖']
    train_x = train_xy.drop(['血糖'], axis=1)
    # 验证集
    val_y = val['血糖']
    val_x = val.drop(['血糖'], axis=1)

    test = pd.read_csv('../data/after_pre_test.csv', encoding='gb2312')

    # cv_params = {'n_estimators': [i * 100 for i in range(1, 10)]}
    # other_params = {'n_estimators': 600, 'max_depth': 2, 'min_samples_split': 4,'learning_rate': 0.025, 'loss': 'ls'}
    # model = ensemble.GradientBoostingRegressor(**other_params)
    # optimized_GDBT = GridSearchCV(
    #     estimator=model,
    #     param_grid=cv_params,
    #     scoring='r2',
    #     cv=5,
    #     verbose=1,
    #     n_jobs=4)
    # optimized_GDBT.fit(train_x, train_y)
    # evalute_result = optimized_GDBT.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GDBT.best_params_))#600
    # print('最佳模型得分:{0}'.format(optimized_GDBT.best_score_))#0.12447971417368059

    # cv_params = {'max_depth': [1, 2, 3, 4, 5, 6], 'min_samples_split': [2, 3, 4, 5, 6]}
    # other_params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 4,'learning_rate': 0.025, 'loss': 'ls'}
    # model = ensemble.GradientBoostingRegressor(**other_params)
    # optimized_GDBT = GridSearchCV(
    #     estimator=model,
    #     param_grid=cv_params,
    #     scoring='r2',
    #     cv=5,
    #     verbose=1,
    #     n_jobs=4)
    # optimized_GDBT.fit(train_x, train_y)
    # evalute_result = optimized_GDBT.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GDBT.best_params_))#{'max_depth': 1, 'min_samples_split': 2}
    # print('最佳模型得分:{0}'.format(optimized_GDBT.best_score_))#0.13675213250100154

    # cv_params = {'learning_rate': [0.025 * i for i in range(1, 5)]}
    # other_params = {'n_estimators': 500, 'max_depth': 1, 'min_samples_split': 2, 'learning_rate': 0.025, 'loss': 'ls'}
    # model = ensemble.GradientBoostingRegressor(**other_params)
    # optimized_GDBT = GridSearchCV(
    #     estimator=model,
    #     param_grid=cv_params,
    #     scoring='r2',
    #     cv=5,
    #     verbose=1,
    #     n_jobs=4)
    # optimized_GDBT.fit(train_x, train_y)
    # evalute_result = optimized_GDBT.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GDBT.best_params_))  # {'learning_rate': 0.075}
    # print('最佳模型得分:{0}'.format(optimized_GDBT.best_score_))  # 0.14851053013686652

    params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 5,
              'learning_rate': 0.025, 'loss': 'ls'}
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(train_x,train_y)
    preds = model.predict(val_x)
    print(metrics.mean_squared_error(val_y, preds)) # 1.693225419888223


