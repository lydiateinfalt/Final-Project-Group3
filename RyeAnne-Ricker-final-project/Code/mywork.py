# Group 3 Preprocessing

# Import libraries
import readdata
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
# import other libraries below:


# Checking to see the number of crashes involving someone 100+ to determine if that's a good
# Age to cap at
print("Crashes involving someone who is 100 or older:")
print(crash[crash.AGE >= 100.0].count()) # Result is 276
print("Crashes involving someone who is less than 0 years of age")
print(crash[crash.AGE< 0.0].count()) # Result is 70

print(crash['AGE'].describe())

###
# RR - Create Feature Matrix and Fill/Drop NANs - 29 lines, 17 I wrote, 12 copied, 12 modified
###
# Create Matrix with only the feature rows we want and the target
print('The list of columns before dropping any include:')
print(crash.columns) # view columns
# drop redundent/unnecessary columns
crash_fm = crash.drop(['PERSONID','FATAL','MAJORINJURY','MINORINJURY'], axis=1)
print('The list of columns after keeping only those that we want to use as features include:')
print(crash_fm.columns) # check the columns again


# Drop drivers under the age of 10 - as that doesn't make sense to have drivers under 10
print('Rows prior to dropping young drivers is:',crash_fm.shape[0])
young_drivers = crash_fm[ (crash_fm['AGE'] < 10) & (crash_fm['PERSONTYPE'] == 'Driver')].index
crash_fm.drop(young_drivers, inplace = True)
print('Rows after dropping young drivers is:', crash_fm.shape[0])

# Fill in missing data
print('The number of empty values per column is')
print(crash_fm.isnull().sum()) # get missing values
crash_fm_age = crash_fm.drop(['AGE'], axis=1) # use to view rows with missing data, except age
null_data = crash_fm_age[crash_fm_age.isnull().any(axis=1)] # view rows with missing data
print(null_data.head(20)) # view the null data rows - Weirdly, they are completely empty - delete 328 empty rows
crash_fm.dropna(subset = ["PERSONTYPE"], inplace=True) # rows missing PERSONTYPE-these will delete all of the empty 333 rows
# crash_fm.dropna(subset = ["AGE"], inplace = True) # Use if Dr Jafari says 28% of missing values is too much and tells us to delet ethem
crash_fm.AGE.fillna(crash_fm.AGE.mean(), inplace=True)  # impute empty age cells with the average value
print('The number of empty values per column after imputation is:')
print(crash_fm.isnull().sum())  # check to make sure there is no longer empty cells - all dealt with
# deal with nan values-there are none
print(crash_fm.groupby(by='IMPAIRED').agg('count'))
print(crash_fm.groupby(by='TICKETISSUED').agg('count'))
print(crash_fm.groupby(by='INVEHICLETYPE').agg('count'))
print(crash_fm.groupby(by='SPEEDING').agg('count'))
print(crash_fm.groupby(by='FATALMAJORINJURIES').agg('count'))

# Label Encoder
ord_enc = OrdinalEncoder()
crash_fm["PERSONTYPE"] = ord_enc.fit_transform(crash_fm[["PERSONTYPE"]])
crash_fm["TICKETISSUED"] = ord_enc.fit_transform(crash_fm[["TICKETISSUED"]])
crash_fm["INVEHICLETYPE"] = ord_enc.fit_transform(crash_fm[["INVEHICLETYPE"]])
crash_fm["SPEEDING"] = ord_enc.fit_transform(crash_fm[["SPEEDING"]])
crash_fm["LICENSEPLATESTATE"] = ord_enc.fit_transform(crash_fm[["LICENSEPLATESTATE"]])
crash_fm["IMPAIRED"] = ord_enc.fit_transform(crash_fm[["IMPAIRED"]])
print('The new label encoded feature matrix is:')
print(crash_fm.head(11))

print('The data types of each feature after reassignment is:')
print(crash_fm.dtypes)

# Data Normalization - for use on Age
cols_to_norm = ['AGE']
crash_fm[cols_to_norm] = StandardScaler().fit_transform(crash_fm[cols_to_norm])
# manual normalization
# normalize age
# mean_age = crash_fm.AGE.mean()
# max_age = crash_fm.AGE.max()
# min_age = crash_fm.AGE.min()
# crash_fm['AGE'] = crash_fm['AGE'].apply(lambda x: (x - mean_age ) / (max_age -min_age)) # normalize
# # Citation: StackOverflow. https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe/28577480
print('Final Check of the final data:')
print(crash_fm.head(20))

# final feature matrix to call
crash_model = crash_fm
crash_model.to_csv("crash_model.csv")
### RR End
### The output is the cleaned, filled, and normalized matrix containing only the features and the target



# Import libraries
import readdata
import researchpy as rp
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import statsmodels

# Arianna Code Start:
crash = readdata.crash
print(crash.shape)

# RR - 30 lines, 15 copied and modified, 15 myself

crash.dropna(subset = ["PERSONTYPE"], inplace=True) # rows missing PERSONTYPE-these will delete all of the empty 328 rows
print('Columns with missing values include:')
print(crash.isnull().sum()) # Age can be missing values as it is not categorical and thus will not have a chi-squared test on it

# This portion determines whether features are independent of crash having a fatality/major injury
# Chi squared tests are performed on each categorical variables to determine independence

# Age - decided not to put into groups and to instead use a quant variable
#crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["AGE"],test= "chi-square",expected_freqs= True,prop= "cell")
#print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Age are:',test_results)
# Vehicle Type
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["INVEHICLETYPE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Vehicle Type are:',test_results)
# Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["IMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired are:',test_results)
# Ticket Issued
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TICKETISSUED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Tickets Issued are:',test_results)
# Lice Plate State
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["LICENSEPLATESTATE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and License Plate State are:',test_results)
# Speeding
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["SPEEDING"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Speeding are:',test_results)
# Person Type
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["PERSONTYPE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Person Type are:',test_results)

# Age is our only quantitative variable - run summary statistics on it
# First remove weird ages (<0 and >100)
age_filter = (crash.AGE > 100.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)
young_drivers = crash[ (crash['AGE'] < 10) & (crash['PERSONTYPE'] == 'Driver')].index
crash.drop(young_drivers, inplace = True)
print('Summary Age statistics are:')
print(crash.AGE.describe())
#print('The mode is:',statistics.mode(crash.AGE))

# Look at distributions of the age of drives for those crashes that result in Fatalities/Major Injuries and Not
sns.set(style="darkgrid")
sns.displot(crash, x="AGE", hue="FATALMAJORINJURIES", kind="kde", multiple="stack",palette="Paired",legend=False)
#sns.displot(data=fmi, x="AGE", color="skyblue", label="Fatal/Major Injury", kde=True)
#sns.displot(data=minor, x="AGE", color="green", label="Minor Injury", kde=True)
plt.legend(labels=['Fatal/Major Injury','Minor Injury'])
plt.title('Distribution of Age By Injury Group')
plt.show()

# Normalize Age so that we can compare, despite the class imbalance
sns.set(style="darkgrid")
sns.displot(crash, x="AGE", hue="FATALMAJORINJURIES", kind="kde", multiple="stack",common_norm=False,palette="Paired",legend=False)
#sns.displot(data=fmi, x="AGE", color="skyblue", label="Fatal/Major Injury", kde=True)
#sns.displot(data=minor, x="AGE", color="green", label="Minor Injury", kde=True)
plt.legend(labels=['Fatal/Major Injury','Minor Injury'])
plt.title('Normalized Age Distributions')
plt.show()

# Run t-test between the ages of those acquiring major injuries/fatalities and minor injuries
fmi = crash.loc[crash['FATALMAJORINJURIES'] == 1.0] # group 1 is those that had a fatal major injury
fmi.dropna(subset = ["AGE"], inplace=True) # drop rows with nan for stats analysis
#print('The number of individuals acquiring a fatality/major injury is', fmi.shape[0])
minor = crash.loc[crash['FATALMAJORINJURIES'] == 0.0] # group 2 is those that did not
minor.dropna(subset = ["AGE"], inplace=True) # drop rows with nan for stats analysis
#print('The number of individuals not acquiring a fatality/major injury is', minor.shape[0])
#print('The proportion of individuals in a DC crash that ends up dead or with a major injury is', fmi.shape[0]/crash.shape[0])

# calculate p-value
t,p = stats.ttest_ind(fmi.AGE, minor.AGE, equal_var=False) # run t-test, do not assume equal variance
print('The p-value of the t-test is:', p)
print('The mean age of those who have a fatality or major injury is', np.mean(fmi.AGE))
print('The standard deviation of fatal/major injury is', np.std(fmi.AGE))
print('The mean age of those who have a minor injury is', np.mean(minor.AGE))
print('The standard deviation of minor injury is', np.std(minor.AGE))

pz = statsmodels.stats.weightstats.ztest( fmi.AGE, minor.AGE)
print('The p value from z test is:',pz) # compare ztest to ttest - 0.014 to 0.015 very similar t distribution tends towards normal at high n


# RR
import webbrowser

from sklearn.tree import export_graphviz

import Preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import plot_tree
from numpy import mean



#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + "C:\Program Files (x86)\Graphviz\bin"
#%%-----------------------------------------------------------------------
#from graphviz.pydotplus import graph_from_dot_data
model = Preprocessing.crash_model


class xgboost:  # class

    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]


    def accuracy(self):  # this makes the model and finds the accuracy, confusion matrix, and prints the decision tree
        # 13 lines of code - 4 copied, 1 modified, 9 myself
        clf = xgb.XGBClassifier(n_estimators=250, # these are the parameters - were adjusted
                                learning_rate=0.01, # tried 0.01,0.05,0.1,0.2
                                max_depth=10, # tried 10, 25, 50
                                min_samples_split=2,
                                min_samples_leaf=1,
                                random_state=100,
                                reg_lambda = 10, # tried 1, 10, 100
                                reg_alpha = 10, # tried 1, 10, 100
                                scale_pos_weight=25 # IMPORTANT! - there is a class imbalance so change the weight on the positives
                                       )
        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3, random_state=10) # split data up
        clf.fit(X_train, y_train) # fit model to training data - non cross-validated results are used for making the confusion matrix
        y_pred = clf.predict(X_test)  # predict testing data - for confusion matrix
        self.roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) # get AUC value
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        # # # cross validate results - 3 copied, 2 modified -these 3 lines can be commented out if you don't want to run CV
        #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        #scores = cross_val_score(clf, self.xtrain, self.ytrain, scoring='roc_auc', cv=cv, n_jobs=-1)
        #print('Mean ROC AUC of cross-validated scores is: %.5f' % mean(scores))

        # Dr. Jafari code - 6 copied, not modified
        # Selecting important features. Lines 33-68 are from Dr. Jafari's code and were updated accordingly
        importances = clf.feature_importances_
        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.xtrain.columns)
        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15, color='r')
        plt.tight_layout()
        #plt.title('Feature Importance', fontsize=20)
        plt.show()

        # Dr. Jafari code - 3 copied, not modified
        conf_matrix = confusion_matrix(y_test, y_pred) # make confusion matrix
        class_names = self.ytrain.unique()  # get the class names
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        # sensitivity and specificity - 4 copied and modified RR
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Specificity : ', specificity)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Sensitivity : ', sensitivity)

        # Dr. Jafari Code - 9 copied, not modified, 1 line myself
        plt.figure(figsize=(5, 5))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title('Extreme Gradient Boosted DT Confusion Matrix')
        plt.tight_layout()
        plt.show()

        # 1 line myself
        return self.roc  # return the accuracy

# 2 lines myself
m = xgboost(model) # put model into class
m.accuracy() # run



# RR - Naive Bayes Model

# import libraries
import Preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean

# 1 myself
model = Preprocessing.crash_model  # call the preprocessed data

# 4 myself
class naivebayes:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]  # all the columns but last are features
        self.ytrain = data.iloc[:,-1]  # last column is the label

    # 1 line of code myself
    def accuracy(self):  # this makes the model and finds the accuracy, and confusion matrix
        # 8 lines - 4 copied not modified, 4 written myself
        clf = GaussianNB()  # model
        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3, random_state=100)  # split data
        clf.fit(X_train, y_train) # fit the model to the training data
        y_pred = clf.predict(X_test)  # predict the testing data
        self.roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) # get AUC value
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        # # cross validate results - 3 copied, 2 modified
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_val_score(clf, self.xtrain, self.ytrain, scoring='roc_auc', cv=cv, n_jobs=-1)
        print('Mean ROC AUC of cross-validated scores is: %.5f' % mean(scores))

        # take from dr. jafari - 3 copied, not modified
        conf_matrix = confusion_matrix(y_test, y_pred)  # make confusion matrix
        class_names = self.ytrain.unique()  # get names of the classes
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        # sensitivity and specificity - 4 copied and modified RR
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Specificity : ', specificity)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Sensitivity : ', sensitivity)

        # taken from Dr. Jafari - 9 copied, not modified, 1 added myself
        plt.figure(figsize=(5, 5))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title('Naive Bayes Confusion Matrix')
        plt.tight_layout()
        plt.show()

        # 1 added myself
        return self.roc  # return the accuracy

# 2 added myself
m = naivebayes(model)  # put model into class
m.accuracy()  # call the code



import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean

# 1 myself
model = Preprocessing.crash_model  # call the preprocessed data


class votingclassifier:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]  # all the columns but last are features
        self.ytrain = data.iloc[:,-1]  # last column is the label

    # 23 lines - 11 myself, 12 copied w 6 modified
    def accuracy(self):  # this makes the model and finds the accuracy, and confusion matrix
        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,
                                                            random_state=100)  # split data
        clf1 = LogisticRegression(class_weight='balanced', penalty='l2',C=.0001)
        # clf2 = GaussianNB()  # model
        clf2 = xgb.XGBClassifier(n_estimators=250, # -I cut the forest because it had little impact on accuracy but saved a lot of time
                                 learning_rate=0.01, # tried 0.01,0.05,0.1,0.2
                                 max_depth=10, # tried 10, 25, 50
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 #warm_start=True,
                                 reg_lambda = 10, # tried 1, 10, 100
                                 reg_alpha = 10, # tried 1, 10, 100
                                 scale_pos_weight=25 # IMPORTANT! - there is a class imbalance so change the weight on the positives
                                        )
        clf3 = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
        eclf = VotingClassifier(estimators = [('Logit', clf1), ('XGB', clf2), ('RF', clf3)],voting = 'hard') # voting classifier

        # Run below for 5-fold Cross Validation - THese 3 lines can be commented out if you don't want to run CV
        for clf, label in zip([clf1, clf2, clf3, eclf], ['Naive Bayes', 'XGBoost', 'Random Forest', 'Ensemble']):
                              scores = cross_val_score(clf, self.xtrain, self.ytrain, scoring='accuracy', cv=5)
        print("5-Fold CV AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


        ### This calculates using a hard voting
        eclf.fit(X_train, y_train)
        y_pred = eclf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("Hard Voting Accuracy:", score)
        # self.roc = roc_auc_score(y_test, eclf.predict_proba(X_test)[:, 1])  # get AUC value
        # print('The AUC of the model is:', self.roc)

        # take from dr. jafari - 3 copied, not modified
        conf_matrix = confusion_matrix(y_test, y_pred)  # make confusion matrix
        class_names = self.ytrain.unique()  # get names of the classes
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        # sensitivity and specificity - 4 copied and modified RR
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Specificity : ', specificity)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Sensitivity : ', sensitivity)

        # taken from Dr. Jafari - 9 copied, not modified, 1 added myself
        plt.figure(figsize=(5, 5))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title('Hard Voting Classifier Confusion Matrix')
        plt.tight_layout()
        plt.show()


        ### Soft voting
        eclf_soft = VotingClassifier(estimators = [('Logit', clf1), ('XGB', clf2), ('RF', clf3)],voting = 'soft') # voting classifier
        eclf_soft.fit(X_train, y_train)
        y_pred_soft = eclf_soft.predict(X_test)
        score_soft = accuracy_score(y_test, y_pred_soft)
        print("Soft Voting Accuracy:", score_soft)
        # self.roc = roc_auc_score(y_test, eclf.predict_proba(X_test)[:, 1])  # get AUC value
        # print('The AUC of the model is:', self.roc)

        # take from dr. jafari - 3 copied, not modified
        conf_matrix_soft = confusion_matrix(y_test, y_pred_soft)  # make confusion matrix
        class_names = self.ytrain.unique()  # get names of the classes
        df_cm_soft = pd.DataFrame(conf_matrix_soft, index=class_names, columns=class_names)

        # sensitivity and specificity - 4 copied and modified RR
        sensitivity_soft = conf_matrix_soft[0, 0] / (conf_matrix_soft[0, 0] + conf_matrix_soft[0, 1])  # calculate sensitivity
        print('Soft Sensitivity : ', sensitivity_soft)
        specificity_soft = conf_matrix_soft[1, 1] / (conf_matrix_soft[1, 0] + conf_matrix_soft[1, 1])  # calculate specificity
        print('Soft Specificity : ', specificity_soft)

        # taken from Dr. Jafari - 9 copied, not modified, 1 added myself
        plt.figure(figsize=(5, 5))
        hm_soft = sns.heatmap(df_cm_soft, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm_soft.columns, xticklabels=df_cm_soft.columns)
        hm_soft.yaxis.set_ticklabels(hm_soft.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm_soft.xaxis.set_ticklabels(hm_soft.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title('Soft Voting Classifier Confusion Matrix')
        plt.tight_layout()
        plt.show()

        # 1 added myself
        # return self.roc  # return the accuracy

# 2 added myself
m = votingclassifier(model)  # put model into class
m.accuracy()  # call the code

# Citation: Ensemble Methods. SKLearn. https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
