import pandas as pd
import itertools
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV


all_features_df= pd.read_pickle('all_features_df.pickle')


# ------------------------- tag normalize -------------------------


tag_cond = [
    (all_features_df['tag'] == 'FALSE'),
    (all_features_df['tag'] == 'TRUE')]
tag_cat = [0,1]
all_features_df['tag_cat'] = np.select(tag_cond, tag_cat)
all_features_df.drop('tag', 1, inplace=True)



df_train = df_train.drop(columns=['Unnamed: 0', 'created_at_x', 'source', 'public_metrics_x',
       'referenced_tweets', 'lang', 'reply_settings', 'id', 'conversation_id',
       'author_id', 'in_reply_to_user_id', 'text', 'geo', 'created_at_y',
       'username', 'name', 'public_metrics_y', 'description', 'follow_rate','following_count' , 'followers_count'])

# -------------------------- Tree ----------------------------------

y_train = df_train["tag_cat"]
X_train = df_train.drop(columns=['tag_cat'])

model=DecisionTreeClassifier(criterion='entropy', max_depth=5)
model.fit(X_train,y_train)
plt.figure(figsize=(25, 10))
plot_tree(model, filled=True, max_depth=4,class_names=['Tagged', 'Not tagged'], fontsize=14, feature_names=X_train.columns)
plt.show()

score = roc_auc_score(y_true=y_train, y_score=model.predict(X_train))
print(score)
score2 = recall_score(y_train, model.predict(X_train))
score3 = precision_score(y_train, model.predict(X_train))
print(score2)
print(score3)


#--------------------random forest-------------------------------



y_train = df_train["tag_cat"]
X_train = df_train.drop(columns=['tag_cat'])

names = ["Linear SVM",
         # "RBF SVM",
         # "Decision Tree",
         "Random Forest",
         "AdaBoost",
         # "Naive Bayes",
         "Logistic Regression",
         "Gradient Boosting Classifier"
         ]
# 'penalty':('l1', 'l2'),
parameters = [{'C': [0.1, 0.5, 1, 10]}
    , {'n_estimators': [50, 100, 150, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [3, 4, 5, 6, 7],
       'criterion': ['gini', 'entropy']}
    , {"n_estimators": [1, 2, 10, 50, 100], 'algorithm': ['SAMME.R', 'SAMME']}
    , {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
    , {'n_estimators': range(20, 101, 20), 'max_depth': range(3, 16, 4), 'min_samples_split': range(2, 10, 2)}
              # ,'max_features':range(7,20,2),'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
              ]

classifiers = [
    LinearSVC(C=0.5, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                           max_depth=7, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                           oob_score=False, random_state=None, verbose=0,
                           warm_start=False),
    AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1.0,
                       n_estimators=10, random_state=None),
    # GaussianNB(),
    LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
                       solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=6,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None,
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)
]
scoring = ['precision', 'recall', 'f1', 'roc_auc']

for name, clf, parameter in zip(names, classifiers, parameters):
    print(name)
    model = model_selection.GridSearchCV(estimator = clf,param_grid = parameter, cv=5)
    model.fit(X_train,y_train)
    print('Best score for data1:', model.best_score_)
    print('Best estimator:',model.best_estimator_)
    #print (model_selection.cross_val_score(clf, X = User_Data_train, y = User_Target_train, cv = 5,scoring=score))
    #clf.fit(X_train, y_train)
    #score = clf.score(X_test, y_test)

