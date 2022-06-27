import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict




os.chdir("C:\\Users\\Niv\\PycharmProjects\\final_project\\Final_project\\pickles")

df= pd.read_pickle('all_features_df_new.pickle')
df.head()
df.info() #get info to check whether dataframe have non values

#print columns in which all rows has the same value
for col in df.columns:
    print(col)
    if len(df[col].unique())==1:
        print(col)

#drop non-relevant columns
df = df.drop(['author_id', 'username', 'URL_bio_count', 'Curse_bio'], axis=1)
df = df.drop(['public_metrics_x', 'public_metrics_y', 'public_metrics', 'number_of_tweets_x', 'description_x', 'name_x', 'description', 'accessible', 'source', 'tweets', 'index'], axis=1)
df = df.dropna()

# ------------ Correlation matrix ------------ #

df1 = df.copy()

cols = df1.columns

plt.figure(figsize = (20, 16), dpi = 100)

corr = df1.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(corr,
            mask = mask,
            cmap = 'autumn',
            vmax=.3,
            annot = True,
            linewidths = 1.5,
            fmt = ".2f",
            alpha = 0.7)

hfont = {'fontname':'monospace'}
plt.xticks(**hfont)
plt.yticks(**hfont)

plt.title('Correlation matrix',
          family = 'monospace',
          fontsize = 20,
          weight = 'semibold',
          color = 'blue')

plt.show()


#-------------------------------------------------------------------------

y = df['tag'] #get y as output model
x = df.drop(['tag'], axis=1) #get X as data to feed to model
#%%
#Plot histogram to represent the distribution of each variable
fig = plt.figure(figsize = (12,20))
ax = fig.gca()
x.hist(ax = ax)


df['tag'].value_counts().plot(kind='bar') #plot bar chart to check imbalance data

y.value_counts().plot.pie(autopct='%.2f') #plot pie chart
plt.show()
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)



# --------------------- split data to train and test --------------------- #

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 0)

#apply standard scalar to scale value from -1 to 1

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)



# ------------------------------------ Create initial models ------------------------------------ #

decisiontree_classifier = DecisionTreeClassifier(criterion='entropy')
decisiontree_classifier.fit(x_train, y_train)

logistic = LogisticRegression()
logistic.fit(x_train, y_train)

fr = RandomForestClassifier(max_depth=9, n_estimators=1000)
fr.fit(x_train, y_train)


print("Accuracy score of decisiontree model is: ", decisiontree_classifier.score(x_test, y_test))
print("Accuracy score of logistic model is: ", logistic.score(x_test, y_test))
print("Accuracy score of random forest model is: ", fr.score(x_test, y_test))


# ------------------ confusion matrix of the initial models ------------------ #


decisiontree_classifier_pred = decisiontree_classifier.predict(x_test)
cm_tree = confusion_matrix(y_test, decisiontree_classifier_pred)

logistic_pred = logistic.predict(x_test)
cm_logistic = confusion_matrix(y_test, logistic_pred)

fr_pred = fr.predict(x_test)
cm_fr = confusion_matrix(y_test, fr_pred)

#confusion matrix decision tree

ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()

#confusion matrix logistic

ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()

#confusion matrix random forest

ax = plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix random forest');
ax.xaxis.set_ticklabels(['True','False']); ax.yaxis.set_ticklabels(['True','False']);
plt.show()

#reports for all 3 models

target_names =['False','True']
print("Report model decision tree\n")
print(classification_report(decisiontree_classifier_pred, y_test, target_names=target_names))


target_names = ['False','True']
print("Report model logistic\n")
print(classification_report(logistic_pred, y_test, target_names=target_names))


target_names = ['False','True']
print("Report model random forest\n")
print(classification_report(fr_pred, y_test, target_names=target_names))





# ------------------------------------ Create processed data models ------------------------------------ #
#using advanced algos to get balanced dataset



# ---------------------------------- Random undersampler ---------------------------------- #

rs=RandomUnderSampler(sampling_strategy=1.0) #Sampling Startegy means the ratio
x=np.array(x)
y=np.array(y)
X_new, y_new = rs.fit_resample(x,y)

x_train, x_test, y_train, y_test = train_test_split(X_new, y_new, test_size = 0.15, random_state = 6)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


decisiontree_classifier = DecisionTreeClassifier()
decisiontree_classifier.fit(x_train, y_train)

logistic = LogisticRegression()
logistic.fit(x_train, y_train)

fr = RandomForestClassifier(max_depth=9, n_estimators=1000)
fr.fit(x_train, y_train)


print("Accuracy score of decisiontree model is: ", decisiontree_classifier.score(x_test, y_test))
print("Accuracy score of logistic model is: ", logistic.score(x_test, y_test))
print("Accuracy score of random forest model is: ", fr.score(x_test, y_test))


# ------------------ confusion matrix of the undersampler models ------------------ #


decisiontree_classifier_pred = decisiontree_classifier.predict(x_test)
cm_tree = confusion_matrix(y_test, decisiontree_classifier_pred)

logistic_pred = logistic.predict(x_test)
cm_logistic = confusion_matrix(y_test, logistic_pred)

fr_pred = fr.predict(x_test)
cm_fr = confusion_matrix(y_test, fr_pred)


#confusion matrix undersampler decision tree

ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Undersampler decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()

#confusion matrix undersampler logistic

ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Undersampler logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()

#confusion matrix undersampler random forest

ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Undersampler random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()


#reports for all 3 models

target_names = ['False','True']
print("Report Undersampler model decision tree\n")
print(classification_report(decisiontree_classifier_pred, y_test, target_names=target_names))

target_names = ['False','True']
print("Report Undersampler model logistic\n")
print(classification_report(logistic_pred, y_test, target_names=target_names))

target_names = ['False','True']
print("Report Undersampler model random forest\n")
print(classification_report(fr_pred, y_test, target_names=target_names))



# ------------------------------------- SMOTE ------------------------------------- #

oversample = SMOTE()
x_new, y_new = oversample.fit_resample(x,y)

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.15, random_state = 6)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


decisiontree_classifier = DecisionTreeClassifier()
decisiontree_classifier.fit(x_train, y_train)

logistic = LogisticRegression()
logistic.fit(x_train, y_train)

fr = RandomForestClassifier(max_depth=9, n_estimators=1000)
fr.fit(x_train, y_train)


print("Accuracy score of decisiontree model is: ", decisiontree_classifier.score(x_test, y_test))
print("Accuracy score of logistic model is: ", logistic.score(x_test, y_test))
print("Accuracy score of random forest model is: ", fr.score(x_test, y_test))



# ------------------ confusion matrix of the SMOTE models ------------------ #


decisiontree_classifier_pred = decisiontree_classifier.predict(x_test)
cm_tree = confusion_matrix(y_test, decisiontree_classifier_pred)

logistic_pred = logistic.predict(x_test)
cm_logistic = confusion_matrix(y_test, logistic_pred)

fr_pred = fr.predict(x_test)
cm_fr = confusion_matrix(y_test, fr_pred)

#confusion matrix smote decision tree

ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix SMOTE decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


#confusion matrix smote logistic

ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix SMOTE logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


#confusion matrix smote random forest

ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix SMOTE random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


#reports for all 3 models

target_names = ['False','True']
print("Report SMOTE model decision tree\n")
print(classification_report(decisiontree_classifier_pred, y_test, target_names=target_names))

target_names = ['False','True']
print("Report SMOTE model logistic\n")
print(classification_report(logistic_pred, y_test, target_names=target_names))

target_names = ['False','True']
print("Report SMOTE model random forest\n")
print(classification_report(fr_pred, y_test, target_names=target_names))








# ------------------------------------------------------------- K-fold ------------------------------------------------------------- #


kf = KFold(n_splits=10, shuffle=True, random_state=6)

scale = StandardScaler()
x = scale.fit_transform(x)


# scores for each model

#decision tree
score = cross_val_score(DecisionTreeClassifier(), x, y, cv= kf)
print(f'Decision tree, Scores for each fold: {score}')
print("Mean score:", score.mean())

#logistic regression
score = cross_val_score(LogisticRegression(), x, y, cv= kf)
print(f'Logistic, Scores for each fold: {score}')
print("Mean score:", score.mean())

#random forest
score = cross_val_score(RandomForestClassifier(), x, y, cv= kf)
print(f'Random forest, Scores for each fold: {score}')
print("Mean score:", score.mean())


# ----------------- confusion matrix ----------------- #

decisiontree_classifier_fold_pred = cross_val_predict(DecisionTreeClassifier(), x, y, cv=kf)
logistic_pred_fold_pred = cross_val_predict(LogisticRegression(), x, y, cv=kf)
fr_fold_pred = cross_val_predict(RandomForestClassifier(), x, y, cv=kf)

cm_tree = confusion_matrix(y, decisiontree_classifier_fold_pred)
cm_logistic = confusion_matrix(y, logistic_pred_fold_pred)
cm_fr = confusion_matrix(y, fr_fold_pred)


# confusion matrix decision tree k-fold
ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix logistic k-fold
ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix random forest k-fold
ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()


#reports for all 3 models

# decision tree k-fold
target_names = ['False','True']
print("Report Kfold model decision tree\n")
print(classification_report(decisiontree_classifier_fold_pred, y, target_names=target_names))

# logistic k-fold
target_names = ['False','True']
print("Report Kfold model logistic\n")
print(classification_report(logistic_pred_fold_pred, y, target_names=target_names))

# random forest k-fold
target_names = ['False','True']
print("Report Kfold model random forest\n")
print(classification_report(fr_fold_pred, y, target_names=target_names))


# ------------------------ UNDERSAMPLING K-fold ------------------------ #

rs=RandomUnderSampler(sampling_strategy=1.0) #Sampling Startegy means the ratio
x_new, y_new = rs.fit_resample(x,y)
scale = StandardScaler()
x_new = scale.fit_transform(x_new)

# scores for each model

#decision tree
score = cross_val_score(DecisionTreeClassifier(), x_new, y_new, cv= kf)
print(f'Decision tree, Scores for each fold: {score}')
print("Mean score:", score.mean())


#logistic regression
score = cross_val_score(LogisticRegression(), x_new, y_new, cv= kf)
print(f'Logistic, Scores for each fold: {score}')
print("Mean score:", score.mean())

#random forest
score = cross_val_score(RandomForestClassifier(), x_new, y_new, cv= kf)
print(f'Random forest, Scores for each fold: {score}')
print("Mean score:", score.mean())


# ----------------- confusion matrix ----------------- #


decisiontree_classifier_fold_pred = cross_val_predict(DecisionTreeClassifier(), x_new, y_new, cv=kf)
logistic_pred_fold_pred = cross_val_predict(LogisticRegression(), x_new, y_new, cv=kf)
fr_fold_pred = cross_val_predict(RandomForestClassifier(), x_new, y_new, cv=kf)

cm_tree = confusion_matrix(y_new, decisiontree_classifier_fold_pred)
cm_logistic = confusion_matrix(y_new, logistic_pred_fold_pred)
cm_fr = confusion_matrix(y_new, fr_fold_pred)


# confusion matrix undersampling decision tree k-fold
ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix undersampling logistic k-fold
ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix undersampling random forest k-fold
ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);



#reports for all 3 models

# decision tree undersampling k-fold
target_names = ['False','True']
print("Report Kfold model decision tree\n")
print(classification_report(decisiontree_classifier_fold_pred, y_new, target_names=target_names))

# logistic undersampling k-fold
target_names = ['False','True']
print("Report Kfold model logistic\n")
print(classification_report(logistic_pred_fold_pred, y_new, target_names=target_names))

# random forest undersampling k-fold
target_names = ['False','True']
print("Report Kfold model random forest\n")
print(classification_report(fr_fold_pred, y_new, target_names=target_names))




# ------------------------ SMOTE K-fold ------------------------ #

oversample = SMOTE()
x_new, y_new = oversample.fit_resample(x,y)
scale = StandardScaler()
x_new = scale.fit_transform(x_new)


# scores for each model

#decision tree
score = cross_val_score(DecisionTreeClassifier(), x_new, y_new, cv= kf)
print(f'Decision tree, Scores for each fold: {score}')
print("Mean score:", score.mean())

#logistic
score = cross_val_score(LogisticRegression(), x_new, y_new, cv= kf)
print(f'Logistic, Scores for each fold: {score}')
print("Mean score:", score.mean())

#random forest
score = cross_val_score(RandomForestClassifier(), x_new, y_new, cv= kf)
print(f'Random forest, Scores for each fold: {score}')
print("Mean score:", score.mean())


# ----------------- confusion matrix ----------------- #

decisiontree_classifier_fold_pred = cross_val_predict(DecisionTreeClassifier(), x_new, y_new, cv=kf)
logistic_pred_fold_pred = cross_val_predict(LogisticRegression(), x_new, y_new, cv=kf)
fr_fold_pred = cross_val_predict(RandomForestClassifier(), x_new, y_new, cv=kf)
cm_tree = confusion_matrix(y_new, decisiontree_classifier_fold_pred)
cm_logistic = confusion_matrix(y_new, logistic_pred_fold_pred)
cm_fr = confusion_matrix(y_new, fr_fold_pred)

# confusion matrix SOMTE decision tree k-fold
ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);

# confusion matrix SOMTE logistic k-fold
ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix SOMTE random forest k-fold
ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


#reports for all 3 models

# decision tree SMOTE k-fold
target_names = ['False','True']
print("Report Kfold model decision tree\n")
print(classification_report(decisiontree_classifier_fold_pred, y_new, target_names=target_names))

# logistic SMOTE k-fold
target_names = ['False','True']
print("Report Kfold model logistic\n")
print(classification_report(logistic_pred_fold_pred, y_new, target_names=target_names))

# random forest SMOTE k-fold
target_names = ['False','True']
print("Report Kfold model random forest\n")
print(classification_report(fr_fold_pred, y_new, target_names=target_names))


#tree plot
plt.figure(figsize=(12, 10))
plot_tree(decisiontree_classifier, filled=True, max_depth=4,class_names=True, fontsize=10)
plt.show()








# ------------------------------------------------------------- K-fold ------------------------------------------------------------- #


kf = KFold(n_splits=3, shuffle=True, random_state=6)

scale = StandardScaler()
x = scale.fit_transform(x)


# scores for each model

#decision tree
score = cross_val_score(DecisionTreeClassifier(), x, y, cv= kf)
print(f'Decision tree, Scores for each fold: {score}')
print("Mean score:", score.mean())

#logistic regression
score = cross_val_score(LogisticRegression(), x, y, cv= kf)
print(f'Logistic, Scores for each fold: {score}')
print("Mean score:", score.mean())

#random forest
score = cross_val_score(RandomForestClassifier(), x, y, cv= kf)
print(f'Random forest, Scores for each fold: {score}')
print("Mean score:", score.mean())


# ----------------- confusion matrix ----------------- #

decisiontree_classifier_fold_pred = cross_val_predict(DecisionTreeClassifier(), x, y, cv=kf)
logistic_pred_fold_pred = cross_val_predict(LogisticRegression(), x, y, cv=kf)
fr_fold_pred = cross_val_predict(RandomForestClassifier(), x, y, cv=kf)

cm_tree = confusion_matrix(y, decisiontree_classifier_fold_pred)
cm_logistic = confusion_matrix(y, logistic_pred_fold_pred)
cm_fr = confusion_matrix(y, fr_fold_pred)


# confusion matrix decision tree k-fold
ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix logistic k-fold
ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix random forest k-fold
ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);
plt.show()


#reports for all 3 models

# decision tree k-fold
target_names = ['False','True']
print("Report Kfold model decision tree\n")
print(classification_report(decisiontree_classifier_fold_pred, y, target_names=target_names))

# logistic k-fold
target_names = ['False','True']
print("Report Kfold model logistic\n")
print(classification_report(logistic_pred_fold_pred, y, target_names=target_names))

# random forest k-fold
target_names = ['False','True']
print("Report Kfold model random forest\n")
print(classification_report(fr_fold_pred, y, target_names=target_names))


# ------------------------ undersampling K-fold ------------------------ #

rs=RandomUnderSampler(sampling_strategy=1.0) #Sampling Startegy means the ratio
x_new, y_new = rs.fit_resample(x,y)
scale = StandardScaler()
x_new = scale.fit_transform(x_new)

# scores for each model

#decision tree
score = cross_val_score(DecisionTreeClassifier(), x_new, y_new, cv= kf)
print(f'Decision tree, Scores for each fold: {score}')
print("Mean score:", score.mean())


#logistic regression
score = cross_val_score(LogisticRegression(), x_new, y_new, cv= kf)
print(f'Logistic, Scores for each fold: {score}')
print("Mean score:", score.mean())

#random forest
score = cross_val_score(RandomForestClassifier(), x_new, y_new, cv= kf)
print(f'Random forest, Scores for each fold: {score}')
print("Mean score:", score.mean())


# ----------------- confusion matrix ----------------- #


decisiontree_classifier_fold_pred = cross_val_predict(DecisionTreeClassifier(), x_new, y_new, cv=kf)
logistic_pred_fold_pred = cross_val_predict(LogisticRegression(), x_new, y_new, cv=kf)
fr_fold_pred = cross_val_predict(RandomForestClassifier(), x_new, y_new, cv=kf)

cm_tree = confusion_matrix(y_new, decisiontree_classifier_fold_pred)
cm_logistic = confusion_matrix(y_new, logistic_pred_fold_pred)
cm_fr = confusion_matrix(y_new, fr_fold_pred)


# confusion matrix undersampling decision tree k-fold
ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix undersampling logistic k-fold
ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix undersampling random forest k-fold
ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);



#reports for all 3 models

# decision tree undersampling k-fold
target_names = ['False','True']
print("Report Kfold model decision tree\n")
print(classification_report(decisiontree_classifier_fold_pred, y_new, target_names=target_names))

# logistic undersampling k-fold
target_names = ['False','True']
print("Report Kfold model logistic\n")
print(classification_report(logistic_pred_fold_pred, y_new, target_names=target_names))

# random forest undersampling k-fold
target_names = ['False','True']
print("Report Kfold model random forest\n")
print(classification_report(fr_fold_pred, y_new, target_names=target_names))




# ------------------------ SMOTE K-fold ------------------------ #

oversample = SMOTE()
x_new, y_new = oversample.fit_resample(x,y)
scale = StandardScaler()
x_new = scale.fit_transform(x_new)


# scores for each model

#decision tree
score = cross_val_score(DecisionTreeClassifier(), x_new, y_new, cv= kf)
print(f'Decision tree, Scores for each fold: {score}')
print("Mean score:", score.mean())

#logistic
score = cross_val_score(LogisticRegression(), x_new, y_new, cv= kf)
print(f'Logistic, Scores for each fold: {score}')
print("Mean score:", score.mean())

#random forest
score = cross_val_score(RandomForestClassifier(), x_new, y_new, cv= kf)
print(f'Random forest, Scores for each fold: {score}')
print("Mean score:", score.mean())


# ----------------- confusion matrix ----------------- #

decisiontree_classifier_fold_pred = cross_val_predict(DecisionTreeClassifier(), x_new, y_new, cv=kf)
logistic_pred_fold_pred = cross_val_predict(LogisticRegression(), x_new, y_new, cv=kf)
fr_fold_pred = cross_val_predict(RandomForestClassifier(), x_new, y_new, cv=kf)
cm_tree = confusion_matrix(y_new, decisiontree_classifier_fold_pred)
cm_logistic = confusion_matrix(y_new, logistic_pred_fold_pred)
cm_fr = confusion_matrix(y_new, fr_fold_pred)

# confusion matrix SOMTE decision tree k-fold
ax= plt.subplot()
sns.heatmap(cm_tree, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold decision tree');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);

# confusion matrix SOMTE logistic k-fold
ax= plt.subplot()
sns.heatmap(cm_logistic, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold logistic');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


# confusion matrix SOMTE random forest k-fold
ax= plt.subplot()
sns.heatmap(cm_fr, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Kfold random forest');
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['False','True']);


#reports for all 3 models

# decision tree SMOTE k-fold
target_names = ['False','True']
print("Report Kfold model decision tree\n")
print(classification_report(decisiontree_classifier_fold_pred, y_new, target_names=target_names))

# logistic SMOTE k-fold
target_names = ['False','True']
print("Report Kfold model logistic\n")
print(classification_report(logistic_pred_fold_pred, y_new, target_names=target_names))

# random forest SMOTE k-fold
target_names = ['False','True']
print("Report Kfold model random forest\n")
print(classification_report(fr_fold_pred, y_new, target_names=target_names))


#tree plot
plt.figure(figsize=(12, 10))
plot_tree(decisiontree_classifier, filled=True, max_depth=4,class_names=True, fontsize=10)
plt.show()