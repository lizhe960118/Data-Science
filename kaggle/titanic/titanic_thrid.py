import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocess import get_top_n_features

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")
titanic_train_data_X = train_data.drop(['Survived'], axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'], axis=1)





