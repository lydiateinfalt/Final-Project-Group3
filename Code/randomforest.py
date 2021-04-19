# Arianna - random forest

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import tree
import pydotplus
import collections
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Getting preprocessed data
import Preprocessing
model = Preprocessing.crash_model

#  Defining random forest algorithm as a class. Lines 20-30 are from Ryeanne's code & were updated accordingly
class randforest:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]

        clf = RandomForestClassifier(n_estimators=100)

        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,
                                                            random_state=100)  # split data up
        clf.fit(X_train, y_train)  # fit model to training data

        # Selecting important features. Lines 33-68 are from Dr. Jafari's code and were updated accordingly
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
        clf_k_features = RandomForestClassifier(n_estimators=100)

        # train the model
        clf_k_features.fit(newX_train, y_train)

        # %%----------------------------------------------------------------------
        # predicton on test using all features
        y_pred = clf.predict(X_test)
        y_pred_score = clf.predict_proba(X_test)

        # prediction on test using k features
        y_pred_k_features = clf_k_features.predict(newX_test)
        y_pred_k_features_score = clf_k_features.predict_proba(newX_test)

        # Testing accuracy. Lines 71-79 were from Reyanne's code and were updated accordingly
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

        # %%-----------------------------------------------------------------------
        # confusion matrix for gini model. Lines 83-125 are from Dr. Jafari's code & were updated accordingly
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

        # Getting number of trees. 4/4 written by Arianna
        print("Number of Trees with all features: ")
        print(len(clf.estimators_)) # 100 trees

        print("Number of Trees with important features: ")
        print(len(clf_k_features.estimators_)) # 100 trees

        # Printing the tree. 4/4 written by Arianna
        plt.figure(figsize=(30,20))
        plt.title("Random Forest Tree No.1")
        tree.plot_tree(clf_k_features.estimators_[0], filled=True)
        plt.show()


# Running the model
m = randforest(model)
