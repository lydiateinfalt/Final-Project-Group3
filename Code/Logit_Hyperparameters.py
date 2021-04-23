# Arianna

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

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(self.xtrain, self.ytrain, test_size=0.3,

                                                            random_state=100)

# Finding optimal hyperparameters. Code is from the following source:
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