# Arianna- Logistic Regression

# Importing libraries
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Getting preprocessed data
import Preprocessing
model = Preprocessing.crash_model

#  Defining random forest algorithm as a class. Lines 21-28 are from Ryeanne's code & were updated accordingly
class logit:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,

                                                            random_state=100)  # split data up
        # creating the classifier object. Lines 31-40 are from Dr. Jafari's code & were updated accordingly
        clf = LogisticRegression(class_weight='balanced')

        # performing training
        clf.fit(X_train, y_train)

        # make predictions
        # predicton on test
        y_pred = clf.predict(X_test)

        y_pred_score = clf.predict_proba(X_test)

        # Testing accuracy. Lines 43-51 were from Reyanne's code and were updated accordingly
        self.roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100)  # get AUC value
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        # confusion matrix. Lines 54-67 were from Dr.Jafari's code & were updated accordingly
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = self.ytrain.unique()

        # sensitivity and specificity - 4 copied and modified RR
        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # calculate sensitivity
        print('Sensitivity : ', sensitivity)
        specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # calculate specificity
        print('Specificity : ', specificity)

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