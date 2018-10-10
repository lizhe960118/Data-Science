#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/2 18:14
@Author  : LI Zhe
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


train_data = pd.read_csv("../data/after_selection_train.csv")
test_data = pd.read_csv("../data/after_selection_test.csv")
titanic_train_data_Y = train_data['Survived']
titanic_train_data_X = train_data.drop(['Survived'], axis=1)
titanic_test_data_X = test_data

# 验证：学习曲线
def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_sizes=np.linspace(.1,
                                                1.0,
                                                5),
                        verbose=0):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters----------
    estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.
    title : string
    Title for the chart.
    X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
    If an integer is passed, it is the number of folds (defaults to 3).
    Specific cross-validation objects can be passed, see
    sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
    Number of jobs to run in parallel (default 1).
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


X = titanic_train_data_X
Y = titanic_train_data_Y

# RandomForest
rf_parameters = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0}

# AdaBoost
ada_parameters = {'n_estimators': 500, 'learning_rate': 0.1}

# ExtraTrees
et_parameters = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0}

# GradientBoosting
gb_parameters = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0}

# DecisionTree
dt_parameters = {'max_depth': 8}

# KNeighbors
knn_parameters = {'n_neighbors': 2}

# SVM
svm_parameters = {'kernel': 'linear', 'C': 0.025}

# XGB
gbm_parameters = {
    'n_estimators': 2000,
    'max_depth': 4,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': -1,
    'scale_pos_weight': 1}

title = "Learning Curves"

plot_learning_curve(
    RandomForestClassifier(
        **rf_parameters),
    title,
    X,
    Y,
    cv=None,
    n_jobs=4,
    train_sizes=[
        50,
        100,
        150,
        200,
        250,
        350,
        400,
        450,
        500])
plt.show()

# 超参数调试
# 特征工程：寻找更好的特征、删去影响较大的冗余特征；
# 模型超参数调试：改进欠拟合或者过拟合的状态；
# 改进模型框架：对于stacking框架的各层模型进行更好的选择