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


## RR - 39 lines, 12 myself, 27 copied, 9 modified

model = Preprocessing.crash_model


class xgboost:  # class

    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]


    def accuracy(self):  # this makes the model and finds the accuracy, confusion matrix, and prints the decision tree
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

        # # # cross validate results - these 3 lines can be commented out if you don't want to run CV
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_val_score(clf, self.xtrain, self.ytrain, scoring='roc_auc', cv=cv, n_jobs=-1)
        print('Mean ROC AUC of cross-validated scores is: %.5f' % mean(scores))

        # Dr. Jafari code
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

        # Dr. Jafari code -
        conf_matrix = confusion_matrix(y_test, y_pred) # make confusion matrix
        class_names = self.ytrain.unique()  # get the class names
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        # sensitivity and specificity -
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Specificity : ', specificity)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Sensitivity : ', sensitivity)

        # Dr. Jafari Code -
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

        return self.roc  # return the accuracy

m = xgboost(model) # put model into class
m.accuracy() # run