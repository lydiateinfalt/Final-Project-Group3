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
        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Sensitivity : ', sensitivity)
        specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Specificity : ', specificity)

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