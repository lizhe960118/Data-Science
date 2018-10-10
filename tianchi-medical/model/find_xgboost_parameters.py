if __name__ == '__main__':
    from sklearn.cross_validation import train_test_split
    import pandas as pd
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.grid_search import GridSearchCV  # Perforing grid search
    from sklearn import metrics

    df = pd.read_csv(
        '../data/after_pre_train_20180928.csv',
        low_memory=False,
        encoding='gbk')

    train_xy, val = train_test_split(df, test_size=0.3, random_state=1)

    y_train = df['血糖']
    x_train = df.drop(['血糖'], axis=1)
    d_train = xgb.DMatrix(x_train, y_train)

    # 训练集
    train_y = train_xy['血糖']
    train_x = train_xy.drop(['血糖'], axis=1)
    # 验证集
    val_y = val['血糖']
    val_x = val.drop(['血糖'], axis=1)


    # cv_params = {'n_estimators': [i * 100 for i in range(1, 10)]}
    # other_params = {
    # 	'learning_rate': 0.1,
    # 	'n_estimators': 500,
    # 	'max_depth': 5,
    # 	'min_child_weight': 1,
    # 	'seed': 0,
    # 	'subsample': 0.8,
    # 	'colsample_bytree': 0.8,
    # 	'gamma': 0,
    # 	'reg_alpha': 0,
    # 	'reg_lambda': 1}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(
    # 	estimator=model,
    # 	param_grid=cv_params,
    # 	scoring='r2',
    # 	cv=5,
    # 	verbose=1,
    # 	n_jobs=4)
    # optimized_GBM.fit(train_x, train_y)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))#73
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))#0.12864353885532503

    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 73, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    # 				'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(
    # 	estimator=model,
    # 	param_grid=cv_params,
    # 	scoring='r2',
    # 	cv=5,
    # 	verbose=1,
    # 	n_jobs=4)
    # optimized_GBM.fit(x_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_)) #{'max_depth': 5, 'min_child_weight': 2}
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # #最佳模型得分:0.14682946638961916

    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # other_params = {
    #     'learning_rate': 0.1,
    #     'n_estimators': 73,
    #     'max_depth': 5,
    #     'min_child_weight': 2,
    #     'seed': 0,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'gamma': 0,
    #     'reg_alpha': 0,
    #     'reg_lambda': 1}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(
    #     estimator=model,
    #     param_grid=cv_params,
    #     scoring='r2',
    #     cv=5,
    #     verbose=1,
    #     n_jobs=4)
    # optimized_GBM.fit(x_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_)) #{'gamma': 0.1}
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_)) #0.1464795160823554

    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # other_params = {
    # 	'learning_rate': 0.1,
    # 	'n_estimators': 73,
    # 	'max_depth': 5,
    # 	'min_child_weight': 2,
    # 	'seed': 0,
    # 	'subsample': 0.8,
    # 	'colsample_bytree': 0.8,
    # 	'gamma': 0.1,
    # 	'reg_alpha': 0,
    # 	'reg_lambda': 1}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(
    # 	estimator=model,
    # 	param_grid=cv_params,
    # 	scoring='r2',
    # 	cv=5,
    # 	verbose=1,
    # 	n_jobs=4)
    # optimized_GBM.fit(x_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_)) #{'colsample_bytree': 0.8, 'subsample': 0.8}
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_)) #0.1464795160823554

    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # other_params = {
    # 	'learning_rate': 0.1,
    # 	'n_estimators': 73,
    # 	'max_depth': 5,
    # 	'min_child_weight': 2,
    # 	'seed': 0,
    # 	'subsample': 0.8,
    # 	'colsample_bytree': 0.8,
    # 	'gamma': 0.1,
    # 	'reg_alpha': 0,
    # 	'reg_lambda': 1}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(
    # 	estimator=model,
    # 	param_grid=cv_params,
    # 	scoring='r2',
    # 	cv=5,
    # 	verbose=1,
    # 	n_jobs=4)
    # optimized_GBM.fit(x_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_)) #{'reg_alpha': 2, 'reg_lambda': 0.05}
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # other_params = {
    # 	'learning_rate': 0.1,
    # 	'n_estimators': 84,
    # 	'max_depth': 3,
    # 	'min_child_weight': 6,
    # 	'seed': 0,
    # 	'subsample': 0.9,
    # 	'colsample_bytree': 0.6,
    # 	'gamma': 0.5,
    # 	'reg_alpha': 2,
    # 	'reg_lambda': 3}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(
    # 	estimator=model,
    # 	param_grid=cv_params,
    # 	scoring='r2',
    # 	cv=5,
    # 	verbose=1,
    # 	n_jobs=4)
    # optimized_GBM.fit(x_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_)) #{'learning_rate': 0.1}
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合
    model = xgb.XGBRegressor(
		learning_rate = 0.1,
		n_estimators = 73,
		max_depth = 5,
		min_child_weight = 2,
		seed = 0,
		subsample = 0.8,
		colsample_bytree = 0.8,
		gamma = 0.1,
		reg_alpha = 2,
		reg_lambda = 0.05)
    # x_train = x_train.as_matrix()
    # y_train = y_train.as_matrix()
    # val_x = val_x.as_matrix()
    # val_y = val_y.as_matrix()
    model.fit(train_x, train_y, verbose=True)
    preds = model.predict(val_x)
    print(metrics.mean_squared_error(val_y, preds)) # 1.7693202248019853

    # model.fit(x_train, y_train)
    # 对测试集进行预测
    # x_test = pd.read_csv('../data/after_pre_test.csv', encoding='gbk')
    # pred = model.predict(x_test)
    # np.savetxt('../data/predict_xgboost.csv', pred, fmt='%f')