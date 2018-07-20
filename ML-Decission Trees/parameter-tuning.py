# -*- coding: utf-8 -*-


import pandas as pd
from sklearn import tree
from sklearn import model_selection
import os

os.chdir("D:/Data Science/Data/")
titanic_train = pd.read_csv("titanic_train.csv")

#explore the dataframe
titanic_train.shape
titanic_train.info()

#convert categorical columns to one-hot encoded columns
titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
X_train.shape
X_train.info()
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier(random_state=2014)
dt_grid = {'max_depth':list(range(3,8)), 'min_samples_split':[2,3,6,7,8], 'criterion':['gini','entropy']}
grid_tree_estimator = model_selection.GridSearchCV(dt, dt_grid, cv=10, n_jobs=5)
grid_tree_estimator.fit(X_train, y_train)
print(grid_tree_estimator.grid_scores_)
print(grid_tree_estimator.best_score_)
print(grid_tree_estimator.best_params_)
print(grid_tree_estimator.score(X_train, y_train))

titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

#fill the missing value for fare column
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Sex','Pclass','Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Name','Age','Ticket','Cabin'], axis=1, inplace=False)
X_test.shape
X_test.info()
titanic_test['Survived'] = grid_tree_estimator.predict(X_test)

titanic_test.to_csv('submission_CVParams.csv', columns=['PassengerId','Survived'],index=False)