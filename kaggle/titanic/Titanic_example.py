import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

train.tail()
test.head()
train.describe()

train.isnull().sum()
#
# title[title.isin(['Capt', 'Col', 'Major', 'Dr', 'Officer', 'Rev'])] = 'Officer'
# deck = full[full.Cabin.isnull()].Cabin.map(lamba x: re.compile('([a-zA-Z]+)').search(x).group())
# checker. = re.compile('([0-9]+)')
# full['Group_num'] = full.Parch + full.SibSp + 1

train = pd.get_dummies(
    train,
    columns=[
        'Embarks',
        'Pclass',
        'Title',
        'Group_size'])
full['NorFare'] = pd.Series(scaler.fit_tramsform(full.Fare.reshape(-1, 1)).reshape(-1). index=full.index)
full.drop(
    labels=[
        'PassengerId',
        'Name',
        'Cabin',
        'Survived',
        'Ticket',
        'Fare'],
    axis=2,
    inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.metric import make_scorer
from sklearn.metric import accuracy_score

scoring = make_scorer(accuracy_score, greater_is_better=True)


def get_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator


from sklearn.neighbors import KNeighborsClassifier
KNN = kNeighborsClassifier(weights='uniform')
parameters = {'n_neighbors': [3, 4, 5], 'p': [1, 2]}
clf_knn = get_model(KNN, parameters, X_train, y_train, scoring)

print(accuracy_score(y_test, clf_knn.predict(X_test)))
plot_learning_curve(clf_knn, 'KNN', X, y, cv=4)

from sklearn.ensemble import VotingClassifier
clf_vc = VotingClassifier(
    estimator=[
        ('xgbl',
         clf_xgbl),
        ('lgl',
         clf_lgl),
        ('svc',
         clf_svc),
        ('rfc1',
         clf_rfc1),
        ('rfc2',
         clf_rfc2),
        ('knn',
         clf_knn)],
    voting='hard',
    weights=[
        4,
        1,
        1,
        1,
        1,
        2])
clf_vc = clf_vc.fit(X_train, y_train)
print(accuracy_score(y_test, clf_vc.predict(X_test)))
plot_learning_curve(ckf_cv, 'Ensemble', X, y, cv=4)


def summission(model, fname, X):
    ans = pd.DataFrame(columns=['PassengerId', 'Survived'])
    ans.PassengerId = PassengerId
    ans.Survived = pd.Series(model.predict(X), index=ans.index)
    ans.to_csv(fname, index=False)
