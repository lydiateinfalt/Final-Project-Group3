# Arianna Dunham
# DATS 6102
# Final Project Individual Code

#------------EDA------------------------------------------

# import libraries
import readdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Getting summary stats for age. 3/3 lines written by Arianna
age_stats = crash['AGE'].describe()
print("The summary statistics for age are: ")
print(age_stats)

# Based on the results of this, this column needs some cleaning (before cleaning max was 237 and min was -7990)
# Deleting rows where age > 122 and where age < 0
# I'm picking 122 as the max because that's the oldest age on record. 2/2 lines written by Arianna
age_filter = (crash.AGE > 122.0) | (crash.AGE < 0.0)
crash = crash.where(~age_filter)

# Printing sum stats again to compare. 2/2 line written by Arianna
print("The summary statistics for age(cleaned) are:")
print(crash['AGE'].describe())

# Dropping all rows where a person is listed as a driver
# and is under the age of 10, which seems to be an unreasonable age. 2/2 written by Arianna
age_filter_2 = (crash.PERSONTYPE == 'Driver') & (crash.AGE < 10.0)
crash = crash.where(~age_filter_2)

# Checking results to ensure above code worked. 2/2 written by Arianna
print("Minimum AGE by person type: ")
print(crash.groupby('PERSONTYPE').agg({'AGE': 'min'}))


# Getting average age and whether or not the accident resulted in major injury or fatality
# to see if there's any discrepancy from total average. 3/3 lines written by Arianna
mf_age = crash.groupby('FATALMAJORINJURIES').agg({'AGE': 'mean'})
print("Average age involved in accidents with fatalities and major injuries: ")
print(mf_age)

# Note about results: about the same

# Getting a histogram of age 5/5 lines written by Arianna
sns.set_palette('icefire')
age_hist = sns.histplot(data=crash, x='AGE', binwidth=5)
age_hist.set_ylabel('Count')
age_hist.set_xlabel('Age')
age_hist.set_title('Age of People Involved in Traffic Accidents')
plt.show()


# Getting a histogram of age and accidents with fatality/major injury. 7/7 lines written by Arianna
mf_filter = crash[crash.FATALMAJORINJURIES.eq(1.0)]
age_mf_hist = sns.histplot(data=mf_filter, x='AGE', binwidth=5)
age_mf_hist.set_ylabel('Count')
age_mf_hist.set_xlabel('Age')
age_mf_hist.set_title('Age of People Involved in Traffic Accidents with Fatalities or Major Injuries')
plt.show()

# Creating a column that lists injury type to use for graphs. 3/3 written by Arianna
crash['INJURYTYPE'] = np.where((crash['FATALMAJORINJURIES'].eq(1.0) | crash['MINORINJURY'].eq('Y')), 'Fatal/Major', 'Minor')
print("New Injury Type column: ")
print(crash['INJURYTYPE'])



# Getting the count of crashes with major/fatal per state. 2/2 by Arianna
states = crash.groupby('LICENSEPLATESTATE').agg({'FATALMAJORINJURIES':'sum'})
print("Crashes per License Plate State:")
print(states)

# Counting total number of crashes from someone with a plate in the DMV and not in the DMV. 13/13 by Arianna
dmv_crash = 0
non_dmv_crash = 0
no_plate = 0
for i in crash['LICENSEPLATESTATE']:
    if i == 'DC':
        dmv_crash += 1
    elif i == 'VA':
        dmv_crash += 1
    elif i == 'MD':
        dmv_crash += 1
    elif i == 'None':
        no_plate =+ 1
    else:
        non_dmv_crash += 1
print("Number of crashes from DMV plate: ") #468593
print(dmv_crash)
print("Number of crashes from non-DMV Plate: ") #125949
print(non_dmv_crash)

# Counting total number of crashes from someone with a plate in the DMV and not in the DMV resulting in major/fatal.
# the first 15 lines were remixed and the remainder were all written by Arianna
# Link for remixed code:
# https://stackoverflow.com/questions/53153703/groupby-count-only-when-a-certain-value-is-present-in-one-of-the-column-in-panda
dc_mf = ((crash['LICENSEPLATESTATE'] == 'DC')
            .groupby([crash['FATALMAJORINJURIES']])
            .sum()
            .astype(int)
            .reset_index(name='count'))


va_mf = ((crash['LICENSEPLATESTATE'] == 'VA')
            .groupby([crash['FATALMAJORINJURIES']])
            .sum()
            .astype(int)
            .reset_index(name='count'))

md_mf = ((crash['LICENSEPLATESTATE'] == 'MD')
            .groupby([crash['FATALMAJORINJURIES']])
            .sum()
            .astype(int)
            .reset_index(name='count'))

dmv_mf = dc_mf.loc[1] + va_mf.loc[1] + md_mf.loc[1]
print("The total number of accidents with DMV plates resulting in fatalities or major injuries is: ")
print(dmv_mf.loc['count']) #14149.0

fatalmajor = 0
for row in crash['FATALMAJORINJURIES']:
    if row == 1:
        fatalmajor += 1
    else:
        continue
non_dmv_mf = fatalmajor - dmv_mf.loc['count'] - no_plate
print("The total number of accident with plates outside of the DMV resulting in fatalities or major injuries is: ")
print(non_dmv_mf) #7596.0


# Checking for duplicates in PersonID column (dropping nulls before
# I do so to ensure those aren't counted. Duplicates indicate the same person
# being involved in multiple accidents. 5/5 written by Arianna
person_dup = crash.dropna().loc[crash['PERSONID'].duplicated()]
print("Total number of duplicate PersonIDs: ")
print(len(person_dup)) # 10
print("List of duplicate PersonIDs: ")
print(person_dup)

# Getting summary stats for duplicate persons. 2/2 written by Arianna
print("Summary statistics of duplicate PersonIds: ")
print(person_dup.describe())
# Notes on results: all resulted in a fatality or major injury. Average age is 34.3
# Although two of the drivers are listed as 0, which can't be correct. We should drop all
# rows where someone is listed as a driver and has an unreasonably young age listed.


# Getting person type counts. 3/3 written by Arianna
person_type = crash.groupby('PERSONTYPE').agg({'PERSONTYPE': 'count'})
print("Person type counts: ")
print(person_type)

# Creating pie chart of person type counts.  6/6 written by Arianna
person_counts = crash.groupby('PERSONTYPE').agg({'PERSONTYPE': 'count'})
person_chart = person_counts.plot.pie(y='PERSONTYPE', labeldistance=None, figsize=(20,20))
plt.ylabel('Count', fontsize=20 )
plt.title('Person Type Counts', fontsize=30)
plt.legend(bbox_to_anchor=(0.85,1.025),loc="upper left", fontsize=12)
plt.show()

# Getting inury counts of person type. 3/3 written by Arianna
person_injury = crash.groupby('PERSONTYPE').agg({'FATALMAJORINJURIES': 'sum', 'MINORINJURY': 'count'})
print("Injury counts by person type: ")
print(person_injury)


# Getting counts for accidents involving speeding. 3/3 written by Arianna
speeding = crash.groupby('SPEEDING').agg({'FATALMAJORINJURIES': 'sum', 'MINORINJURY': 'count'})
print('Injury counts for accidents involving speeding: ')
print(speeding)

# Getting counts for accidents involving impairment. 3/3 written by Arianna
impaired = crash.groupby('IMPAIRED').agg({'FATALMAJORINJURIES': 'sum', 'MINORINJURY': 'count'})
print('Injury counts for accidents involving speeding: ')
print(impaired)


# Getting counts for accidents where ticket was issued. 3/3 written by Arianna
ticket = crash.groupby('TICKETISSUED').agg({'FATALMAJORINJURIES': 'sum', 'MINORINJURY': 'count'})
print('Injury counts for accidents where a ticket was issued: ')
print(ticket)


# Getting injury counts based on vehicle type. 3/3 written by Arianna
vehicle = crash.groupby('INVEHICLETYPE').agg({'FATALMAJORINJURIES': 'sum', 'MINORINJURY': 'count'})
print("Injury counts by vehicle type: ")
print(vehicle)

# Getting pie chart to show vehicle type break down. 6/6 written by Arianna
vehicle_counts = crash.groupby('INVEHICLETYPE').agg({'INVEHICLETYPE': 'count'})
vehicle_chart = vehicle_counts.plot.pie(y='INVEHICLETYPE', labeldistance=None, figsize=(20,20))
plt.ylabel('Vehicle Type', fontsize=20)
plt.title('Types of Vehicles Involved in Accidents', fontsize=30)
plt.legend(bbox_to_anchor=(0.85,1.025),loc="upper left", fontsize=12)
plt.show()


# Getting a list of the top 10 most dangerous vehicles
# (Vehicles with most fatalities and major injuries). 4/4 written by Arianna
vehicle_mf = crash.groupby('INVEHICLETYPE').agg({'FATALMAJORINJURIES': 'sum'})
max_vehicle = vehicle_mf.nlargest(10, 'FATALMAJORINJURIES')
print("Top 10 dangerous vehicles: ")
print(max_vehicle)

# creating a bar graph depicting top 10 most dangerous vehicles. 6/6 written by Arianna
max_vehicle.plot.bar()
plt.ylabel('Fatality and Major Injury Count')
plt.xlabel('Vehicle Type')
plt.title('Top 10 Most Common Vehicle Types Associated \n With Major Injuries and Fatalities')
plt.tight_layout()
plt.show()


#------------Logit Hyperparameter Optimization-----------------------

# NOTE: This takes several hours to run. The output has been commented at the end of the script

# Importing libraries
import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Getting preprocessed data. 2/2 written by Arianna
import Preprocessing
model = Preprocessing.crash_model

# Setting features/target 2/2 written by Arianna
xtrain = data.iloc[:,:-1]
ytrain = data.iloc[:,-1]

# Splitting the data. 2/2 written by Arianna.
X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,

                                                            random_state=100)

# Finding optimal hyperparameters. All of the code below is from the following source:
# https: // www.dezyre.com / recipes / optimize - hyper - parameters - of - logistic - regression - model - using - grid - search - in -python
std_slc = StandardScaler()
pca = decomposition.PCA()
logistic_Reg = linear_model.LogisticRegression()
pipe = Pipeline(steps=[('std_slc', std_slc),
                       ('pca', pca),
                       ('logistic_Reg', logistic_Reg)])
n_components = list(range(1, self.xtrain.shape[1] + 1, 1))
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']
parameters = dict(pca__n_components=n_components,
                  logistic_Reg__C=C,
                  logistic_Reg__penalty=penalty)

clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(self.xtrain, self.ytrain)

print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print();
print(clf_GS.best_estimator_.get_params()['logistic_Reg'])

# Output:
# Penalty:L2
# C .0001
# Numberofcomponents:1
# LogisticRegression(C=.0001)


#------------Logit-----------------------


# Importing libraries
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Getting preprocessed data. 2/2 written by Arianna
import Preprocessing
model = Preprocessing.crash_model

#  Defining random forest algorithm as a class. Lines 21-28 are from RyeAnne's code & were updated by Arianna accordingly
class logit:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,

                                                            random_state=100)  # split data up
        # creating the classifier object. Lines 31-41 are from Dr. Jafari's code & were updated by Arianna accordingly
        clf = LogisticRegression(class_weight='balanced', penalty='l2',C=.0001, random_state=100) # Hyperparameters set based on results from Logit_HyperParameter file


        # performing training
        clf.fit(X_train, y_train)

        # make predictions
        # predicton on test
        y_pred = clf.predict(X_test)

        y_pred_score = clf.predict_proba(X_test)

        # Testing accuracy. Lines 43-51 are from ReyAnne's code and were updated by Arianna accordingly
        self.roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100)  # get AUC value
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        # confusion matrix. Lines 50-74 are from Dr.Jafari's code & were updated accordingly
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = self.ytrain.unique()

        # sensitivity and specificity - 4 copied and modified RR
        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Sensitivity : ', sensitivity)
        specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Specificity : ', specificity)

        # Cross validation. 2/2 lines copied from Internet and modified by Arianna
        scores = cross_val_score(clf, self.xtrain, self.ytrain, cv=5)
        print("Cross-Validation Accuracy Scores: ", scores)

        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        plt.figure(figsize=(5, 5))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title("Logistic Regression Confussion Matrix")
        plt.tight_layout()
        plt.show()

# Running the model
m = logit(model)

#------------Random Forest----------------------

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn import tree
#import pydotplus
import collections
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Getting preprocessed data. 2/2 written by Arianna
import Preprocessing
model = Preprocessing.crash_model

#  Defining random forest algorithm as a class. Lines 25-34 are from RyeAnne's code & were updated by Arianna accordingly
class randforest:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=100)

        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,
                                                            random_state=100)  # split data up
        clf.fit(X_train, y_train)  # fit model to training data

        # Selecting important features. Lines 37-74 are from Dr. Jafari's code and were updated by Arianna accordingly
        importances = clf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.xtrain.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        # make the bar Plot from f_importances
        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(20, 10), rot=90, fontsize=15)

        # show the plot
        plt.title("Important Features", fontsize=30)
        plt.xlabel("Feature Names", fontsize=20)
        plt.gcf().subplots_adjust(bottom=0.35)
        plt.show()

        # Generating model with important features
        self.Xtrain = data.iloc[:, :-4]
        newX_train, newX_test, y_train, y_test = train_test_split(self.Xtrain, self.ytrain, test_size=0.3,
                                                            random_state=100)

        # %%-----------------------------------------------------------------------
        # perform training with random forest with k columns
        # specify random forest classifier
        clf_k_features = RandomForestClassifier(n_estimators=100, class_weight = 'balanced_subsample')

        # train the model
        clf_k_features.fit(newX_train, y_train)

        # %%----------------------------------------------------------------------
        # predicton on test using all features
        y_pred = clf.predict(X_test)
        y_pred_score = clf.predict_proba(X_test)

        # prediction on test using k features
        y_pred_k_features = clf_k_features.predict(newX_test)
        y_pred_k_features_score = clf_k_features.predict_proba(newX_test)

        # Testing accuracy. Lines 77-87 were from ReyAnne's code and were updated by Arianna accordingly
        self.roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100) # get AUC value
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print("Results using all features: ")
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        self.roc = roc_auc_score(y_test, y_pred_k_features_score[:, 1]) * 100  # get AUC value
        self.acc = accuracy_score(y_test, y_pred_k_features) * 100  # get the accuracy of the model
        print("Results using important features: ")
        print('The AUC of the important features model is:', self.roc)
        print('The classification accuracy for the important features is:', self.acc)

        # # # Cross validation - This takes a couple minutes to run and can be commented out to shorten run time
        # Output has been commented below
        # # 2/2 lines copied from Internet and modified by Arianna
        scores = cross_val_score(clf, self.xtrain, self.ytrain, cv=5)
        print("Cross-Validation Accuracy Scores: ", scores)
        # Output: Cross-Validation Accuracy Scores:  [0.79232041 0.76155973 0.58281704 0.78264903 0.78118765]

        # %%-----------------------------------------------------------------------
        # confusion matrix for gini model. Lines 98-152 are from Dr. Jafari's code & were updated by Arianna accordingly
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = self.ytrain.unique()
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        # sensitivity and specificity - 4 copied and modified RR
        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Sensitivity : ', sensitivity)
        specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Specificity : ', specificity)


        plt.figure(figsize=(5, 5))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)

        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title('Random Forest Confusion Matrix Gini Model')
        # Show heat map
        plt.tight_layout()

        # %%-----------------------------------------------------------------------

        # confusion matrix for entropy model
        conf_matrix = confusion_matrix(y_test, y_pred_k_features)
        class_names = self.ytrain.unique()
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        plt.figure(figsize=(5, 5))

        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)

        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.title('Random Forest Confusion Matrix Entropy Model')
        # Show heat map
        plt.tight_layout()
        plt.show()

        # Printing first tree from all features. 4/4 written by Arianna
        plt.figure(figsize=(15, 10))
        plt.title("All Features Random Forest Tree No.1")
        tree.plot_tree(clf.estimators_[0], filled=True, max_depth=3)
        plt.show()

        # Printing first tree from important features. 4/4 written by Arianna
        plt.figure(figsize=(15,10))
        plt.title("Important Features Random Forest Tree No.1")
        tree.plot_tree(clf_k_features.estimators_[0], filled=True, max_depth=3)
        plt.show()


# Running the model
m = randforest(model)
