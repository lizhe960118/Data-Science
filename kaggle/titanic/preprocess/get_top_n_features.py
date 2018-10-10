# 模型融合及测试

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_top_n_features(
        titanic_train_data_X,
        titanic_train_data_Y,
        top_n_features):


    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {
        'n_estimators': [500],
        'min_samples_split': [
            2,
            3],
        'max_depth': [20]}

    rf_grid = model_selection.GridSearchCV(
        rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' +
          str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame(
        {
            'feature': list(titanic_train_data_X),
            'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance',
        ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(
        ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' +
          str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame(
        {
            'feature': list(titanic_train_data_X),
            'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance',
        ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {
        'n_estimators': [500],
        'min_samples_split': [
            3,
            4],
        'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(
        et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' +
          str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame(
        {
            'feature': list(titanic_train_data_X),
            'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance',
        ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {
        'n_estimators': [500],
        'learning_rate': [
            0.01,
            0.1],
        'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(
        gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' +
          str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame(
        {
            'feature': list(titanic_train_data_X),
            'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance',
        ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(
        dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' +
          str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame(
        {
            'feature': list(titanic_train_data_X),
            'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance',
        ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat([features_top_n_rf,
                                features_top_n_ada,
                                features_top_n_et,
                                features_top_n_gb,
                                features_top_n_dt],
                               ignore_index=True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf,
                                     feature_imp_sorted_ada,
                                     feature_imp_sorted_et,
                                     feature_imp_sorted_gb,
                                     feature_imp_sorted_dt],
                                    ignore_index=True)

    return features_top_n, features_importance

if __name__ == '__main__':
    train_data = pd.read_csv("../data/after_pre_train_20181002.csv")
    test_data = pd.read_csv("../data/after_pre_test_20181002.csv")
    titanic_train_data_Y = train_data['Survived']

    train_data.drop(['Survived'], axis=1,inplace=True)
    titanic_train_data_X = train_data

    titanic_test_data_X = test_data

    feature_to_pick = 30
    feature_top_n, feature_importance = get_top_n_features(
        titanic_train_data_X, titanic_train_data_Y, feature_to_pick)

    # feature_top_n.to_csv("data/features_top_n.csv", index= False)
    # feature_importance.to_csv('data/features_improtance.csv', index=False)

    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

    after_selection_train = pd.concat(
        [titanic_train_data_X, titanic_train_data_Y], axis=1)

    after_selection_train.to_csv("../data/after_selection_train.csv", index=False)
    titanic_test_data_X.to_csv("../data/after_selection_test.csv", index=False)

    rf_feature_imp = feature_importance[:10]
    Ada_feature_imp = feature_importance[35:35 + 10].reset_index(drop=True)

    # make importances relative to max importance
    rf_feature_importance = 100.0 * \
                            (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())
    Ada_feature_importance = 100.0 * \
                             (Ada_feature_imp['importance'] / Ada_feature_imp['importance'].max())

    # Get the indexes of all features over the importance threshold
    rf_important_idx = np.where(rf_feature_importance)[0]
    Ada_important_idx = np.where(Ada_feature_importance)[0]

    # Adapted from Gradient Boosting regression
    pos = np.arange(rf_important_idx.shape[0]) + .5
    plt.figure(1, figsize=(18, 8))
    plt.subplot(121)
    plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])
    plt.yticks(pos, rf_feature_imp['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('RandomForest Feature Importance')
    plt.subplot(122)
    plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])
    plt.yticks(pos, Ada_feature_imp['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('AdaBoost Feature Importance')
    plt.show()