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


model = Preprocessing.crash_model


class gboost:  # class
    def __init__(self, data):  # to call self
        # data is the entire data matrix
        self.x = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]


    def accuracy(self):  # this makes the model and finds the accuracy, confusion matrix, and prints the decision tree
        s = StratifiedShuffleSplit(n_splits=5, random_state=10)  # way to split data
        scoring = {"Accuracy": 'accuracy', "AUC": 'roc_auc'} # have accuracy for both AUC and overall accuracy used
        params = {'n_estimators': [50,100,500],
                    'learning_rate': [0.001, 0.01, 0.1, 1],
                    'max_depth':[3,4,5,6],
                    'min_samples_split': [2,4,6],
                    'min_samples_leaf': [1,2]
                 }
        gb_search = GridSearchCV(GradientBoostingClassifier(random_state = 10),  # make grid to find highest accuracy
                                  params, cv=s.split(self.x, self.y), scoring=scoring, refit='AUC')  # use borth scoring but refit by the user choise
        gb_search.fit(self.x, self.y)  # fit model
        print('The best parameters of the model are:', gb_search.best_params_)

        clf = GradientBoostingClassifier(params = gb_search.best_params_, scoring = 'roc_auc')
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        print("Accuracy of the model: ", self.acc)

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = self.y.unique()
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

        # dot_data = export_graphviz(clf, filled=True, rounded=True, class_names=class_names,
        #                            feature_names=self.x.iloc[:, :].columns, out_file=None)
        #
        # graph = graph_from_dot_data(dot_data)
        # graph.write_pdf("decision_tree_gini.pdf")
        # webbrowser.open_new(r'decision_tree_gini.pdf')

        return gb_search.best_score_ # return the accuracy

m = gboost(model)  # set your roman numeral

