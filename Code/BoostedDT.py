# RR
import Preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


model = Preprocessing.crash_model


class gboost:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.xtrain = data.iloc[:,:-1]
        self.ytrain = data.iloc[:,-1]


    def accuracy(self):  # this makes the model and finds the accuracy, confusion matrix, and prints the decision tree

        clf = GradientBoostingClassifier(n_estimators=50,
                                        learning_rate=0.05,
                                        max_depth=7,
                                        min_samples_split=2,
                                        min_samples_leaf=2,
                                        random_state = 50)
        X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3, random_state=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print('The AUC of the model is:', self.roc)
        print('The classification accuracy is:', self.acc)

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = self.ytrain.unique()
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        plt.figure(figsize=(5, 5))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.tight_layout()
        plt.show()

        dot_data = export_graphviz(clf, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.x.iloc[:, :].columns, out_file=None)

        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_gini.pdf")
        webbrowser.open_new(r'decision_tree_gini.pdf')

        return self.acc# return the accuracy

m = gboost(model)  # set your roman numeral
m.accuracy()