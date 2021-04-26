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

# RR - 36 lines, 10 myself, 26 copied,9 modified


model = Preprocessing.crash_model  # call the preprocessed data


class naivebayes:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]  # all the columns but last are features
        self.ytrain = data.iloc[:,-1]  # last column is the label

    def accuracy(self):  # this makes the model and finds the accuracy, and confusion matrix
        clf = GaussianNB()  # model
        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3, random_state=100)  # split data
        clf.fit(X_train, y_train) # fit the model to the training data
        y_pred = clf.predict(X_test)  # predict the testing data
        self.roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) # get AUC value
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        # # cross validate results -
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_val_score(clf, self.xtrain, self.ytrain, scoring='roc_auc', cv=cv, n_jobs=-1)
        print('Mean ROC AUC of cross-validated scores is: %.5f' % mean(scores))

        # take from dr. jafari -
        conf_matrix = confusion_matrix(y_test, y_pred)  # make confusion matrix
        class_names = self.ytrain.unique()  # get names of the classes
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        # sensitivity and specificity
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Specificity : ', specificity)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Sensitivity : ', sensitivity)

        # taken from Dr. Jafari
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

        return self.roc  # return the accuracy

m = naivebayes(model)  # put model into class
m.accuracy()  # call the code