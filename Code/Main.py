# Group 3 DATS 6103
# DATS 6103 Spring21
# Arianna Dunham, RyeAnne Ricker, Lydia Teinfalt
#%%-----------------------------------------------------------------------

import sys
#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QTableWidget, QTableWidgetItem)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
import scipy
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import seaborn as sns
import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt



#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------

#::--------------------------------
# Default font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'


class Logit(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Logit, self).__init__()
        self.Title = "Logit"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Logit Features')
        self.groupBox1Layout = QGridLayout()  # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)

        self.btnExecute = QPushButton("Execute Logit")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)
        self.groupBox1Layout.addWidget(self.feature6, 3, 0)
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()

        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)
        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------
        #self.layout.addWidget(self.groupBox2, 1, 0)
        #self.layout.addWidget(self.groupBox1, 0, 0)
        #self.layout.addWidget(self.groupBoxG1, 0, 1)

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG2,1,1)
        #self.layout.addWidget(self.groupBoxG3,0,2)
        #self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Logit
        We populate the dashboard using the parameters chosen by the user
        The parameters are processed to execute in the logit algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = crash_model[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[6]]],axis=1)
        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        #self.ax2.clear()
        #self.ax3.clear()
        #self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        X_dt =  self.list_corr_features
        y_dt = crash_model["FATALMAJORINJURIES"]

        class_le = LabelEncoder()

        # fit and transform the class
        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.3, random_state=100)
        self.clf = LogisticRegression(class_weight='balanced', penalty='l2',
                                 C=.0001)  # Hyperparameters set based on results from Logit_HyperParameter file

        # performing training
        self.clf.fit(X_train, y_train)

        # make predictions
        # prediction on test
        y_pred = self.clf.predict(X_test)

        y_pred_score = self.clf.predict_proba(X_test)

        # Testing accuracy.
#        roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100)  # get AUC value
#        acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model

        # confusion matrix.
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = np.unique(y_train)

        # classification report
        self.class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.class_rep)
        self.roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100)  # get AUC value
        # accuracy score
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        self.txtAccuracy.setText(str(self.acc))

        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names)
        self.ax1.set_xticklabels(class_names, rotation=90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------

        probs = self.clf.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        lw = 2
        self.ax2.plot(fpr, tpr, color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Logit Model')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the crash dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)
        self.groupBox1Layout.addWidget(self.feature6, 3, 0)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : PR Curve
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('PR Curve')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2)
        self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We populate the dashboard using the parameters chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = crash_model[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[6]]],axis=1)

        vtest_per = float(self.txtPercentTest.text())

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        #self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt =  self.list_corr_features
        y_dt = crash_model["FATALMAJORINJURIES"]

        class_le = LabelEncoder()
        # fit and transform the class
        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)

        #specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, class_weight = 'balanced_subsample')

        # perform training
        self.clf_rf.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # prediction on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # classification report
        self.class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.class_rep)
        self.roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100) # get AUC value
        # accuracy score
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        self.txtAccuracy.setText(str(self.acc))

        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = np.unique(y_train)

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------

        probs = self.clf_rf.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        lw = 2
        self.ax2.plot(fpr, tpr, color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Random Forest')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()
        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.list_corr_features.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - PR Curve
        #::-----------------------------------------------------
        # Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

        precision, recall, _ = precision_recall_curve(y_test, preds)
        auc1 = auc(recall, precision)
        self.ax4.plot(recall, precision, color='darkorange',
                      lw=lw, label='PR auc (area = %0.2f)' % auc1 )
        self.ax4.set_xlabel('Recall')
        self.ax4.set_ylabel('Precision')
        self.ax4.set_title('Precision Recall Curve')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph  4 - PR Curve
        #::-----------------------------

class Boosted(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of XGBoost DT using the crash dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Boosted, self).__init__()
        self.Title = "XGBoost DT"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('XGBoost Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute XGBoost")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)
        self.groupBox1Layout.addWidget(self.feature6, 3, 0)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : PR Curve
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('PR Curve')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2)
        self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We populate the dashboard using the parameters chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = crash_model[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = crash_model[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, crash_model[features_list[6]]],axis=1)

        vtest_per = float(self.txtPercentTest.text())

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        #self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt =  self.list_corr_features
        y_dt = crash_model["FATALMAJORINJURIES"]

        class_le = LabelEncoder()
        # fit and transform the class
        y_dt = class_le.fit_transform(y_dt)


        #specify XGBoost forest  - RyeAnne's Code
        self.xgboost = xgb.XGBClassifier(n_estimators=250, # these are the parameters - were adjusted
                                learning_rate=0.01, # tried 0.01,0.05,0.1,0.2
                                max_depth=10, # tried 10, 25, 50
                                min_samples_split=2,
                                min_samples_leaf=1,
                                random_state=100,
                                reg_lambda = 10, # tried 1, 10, 100
                                reg_alpha = 10, # tried 1, 10, 100
                                scale_pos_weight=25 # IMPORTANT! - there is a class imbalance so change the weight on the positives
                                       )
        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=10)
        # perform training
        self.xgboost.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # prediction on test using all features
        y_pred = self.xgboost.predict(X_test)
        y_pred_score = self.xgboost.predict_proba(X_test)


        # confusion matrix for XGBoost
        conf_matrix = confusion_matrix(y_test, y_pred)

        # classification report
        self.class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.class_rep)
        self.roc = roc_auc_score(y_test, y_pred_score[:, 1] * 100) # get AUC value
        # accuracy score
        self.acc = accuracy_score(y_test, y_pred) * 100  # get the accuracy of the model
        self.txtAccuracy.setText(str(self.acc))

        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = np.unique(y_train)

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.xgboost.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------

        probs = self.xgboost.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)


        lw = 2
        self.ax2.plot(fpr, tpr, color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Random Forest')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()
        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.xgboost.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.list_corr_features.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - PR Curve
        #::-----------------------------------------------------
        # Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

        precision, recall, _ = precision_recall_curve(y_test, preds)
        auc1 = auc(recall, precision)
        self.ax4.plot(recall, precision, color='darkorange',
                      lw=lw, label='PR auc (area = %0.2f)' % auc1)
        self.ax4.set_xlabel('Recall')
        self.ax4.set_ylabel('Precision')
        self.ax4.set_title('Precision Recall Curve')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - PR curve
        #::-----------------------------


class Crash_Graphs(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the happiness score
    # methods
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a dotplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(Crash_Graphs, self).__init__()

        self.Title = "Crash DC Features"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["IMPAIRED", "SPEEDING", "TICKETISSUED", "PERSONTYPE"])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.checkbox1 = QCheckBox('Fatal/MajorInjuries', self)
        self.checkbox1.stateChanged.connect(self.update)

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Index for subplots"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.checkbox1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        colors = ["b", "r", "g", "y", "k", "c"]
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        if self.checkbox1.isChecked():
            df = fatal_crash
        else:
            df = crash

        df1 = df.groupby(cat1).agg({cat1: 'count'})
        df = pd.DataFrame(data=df1)

        x = df.index.tolist()
        y = df.iloc[:, 0].tolist()
        self.ax1.bar(x,y)

        vtitle = "DC Crash " + cat1
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("DC Crash")
        self.ax1.set_ylabel(cat1)
        self.ax1.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=7, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'DC Crashes Fatal/Major Injuries'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class Canvas(QMainWindow):
    #::----------------------------------
    # Creates a canvas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(Canvas, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'DC Crashes Fatal/Major Injuries'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.s = PlotCanvas(self, width=5, height=4)
        self.s.move(0, 30)

class DataTable(QMainWindow):
        def __init__(self, parent=None):
            super(DataTable, self).__init__(parent)

            self.left = 500
            self.top = 500
            self.Title = 'DC Crash Data'
            self.width = 1000
            self.height = 500
            self.initUI()

        def initUI(self):
            self.t = QTableWidget()
            self.t.setStyleSheet(font_size_window)
            self.t.setGeometry(self.left, self.top, self.width, self.height)
            self.t.setWindowTitle(self.Title)
            self.t.move(0, 30)


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'DC Crash Data'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        DataMenu = mainMenu.addMenu('Data')
        EDAMenu = mainMenu.addMenu('EDA')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), '&Quit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Quit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # Dataset
        #
        #
        #
        #
        #::----------------------------------------

        DataButton = QAction(QIcon(),'Sample ', self)
        DataButton.setStatusTip('Sample data')
        DataButton.triggered.connect(self.Data1)
        DataMenu.addAction(DataButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # 1. Histogram of Age
        #
        #
        #::----------------------------------------

        EDA1Button = QAction(QIcon(),'Age Histogram', self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon(),'Age Scatter', self)
        EDA2Button.setStatusTip('Person Type')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA3Button = QAction(QIcon(),'Crash graphs', self)
        EDA3Button.setStatusTip('Person Type')
        EDA3Button.triggered.connect(self.EDA3)
        EDAMenu.addAction(EDA3Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are three models
        #       Logit
        #       Random Forest
        #       XGBoost
        #
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button =  QAction(QIcon(), 'Logit', self)
        MLModel1Button.setStatusTip('Logit')
        MLModel1Button.triggered.connect(self.LG)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)
        self.dialogs = list()

        #::------------------------------------------------------
        # XGBoost DT
        #::------------------------------------------------------
        MLModel3Button = QAction(QIcon(), 'XGBoost DT', self)
        MLModel3Button.setStatusTip('XGBoost DT ')
        MLModel3Button.triggered.connect(self.XGBoost)

        MLModelMenu.addAction(MLModel3Button)
        self.dialogs = list()

    # Reference: https://pythonspot.com/pyqt5-table/
    def Data1(self):
        dialog = DataTable(self)
        #dialog.t.setWindowTitle('DC Crash Data')
        # Set number of rows
        dialog.t.setRowCount(top_10.shape[0])
        num_cols = len(columns_list)
        dialog.t.setColumnCount(num_cols)

        # Set column headings
        for i in range(0,num_cols):
            dialog.t.setItem(0, i, QTableWidgetItem(columns_list[i]))

        # Read in data
        for i in range(0,5):
            for k in range(1, num_cols+1):
                dialog.t.setItem(i+1,k-1,QTableWidgetItem(str(top_10.iloc[i][k])))

        dialog.t.horizontalHeader().setStretchLastSection(True)
        dialog.t.verticalHeader().setStretchLastSection(True)
        dialog.t.show()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains Age
        # X was populated in the method crash_data
        # at the start of the application
        #::------------------------------------------------------
        dialog = CanvasWindow(self)
        dialog.m.plot()
        dialog.m.ax.hist(X, bins=25, facecolor="skyblue", alpha=0.5, edgecolor="black", linewidth = 1.1)
        dialog.m.ax.set_title('Age of People Involved in Traffic Accidents with Fatalities or Major Injuries')
        dialog.m.ax.set_xlabel("Age of Person(s) Involved")
        dialog.m.ax.set_ylabel("Number Fatal/Major Injuries Accidents")
        dialog.m.ax.axis([0, 100, 0, 1400])
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This function create bar plot of persons injured by transportation mode
        #::---------------------------------------------------------
        dialog1 = Canvas(self)
        dialog1.s.plot()
        #x=fatal_mode.COUNT.tolist()
        #y=fatal_mode.index.tolist()

        dialog1.s.ax.scatter(crash['FATALMAJORINJURIES'],crash['AGE'], alpha=0.01)
        #dialog1.s.ax.set_title('Fatal/Major Injuries by Transportation Mode')
        #dialog1.s.ax.set_ylabel("")
        #fatal_mode = fatal_crash.groupby('PERSONTYPE').agg({'PERSONTYPE': 'count'})
        #fatal_mode = pd.DataFrame(data=fatal_mode)
        #fatal_mode.rename(columns={'PERSONTYPE': 'COUNT'}, inplace=True)
        #fatal_mode.sort_values(by=['COUNT'], inplace=True)
        dialog1.s.ax.axis([-0.25, 1.25, -5, 110])
        dialog1.s.ax.set_ylabel("Age of Person(s) Involved")
        dialog1.s.ax.set_xlabel("Minor Injuries (0) versus Major Injuries/Fatal (1)")
        dialog1.s.ax.set_title('Age x Minor Major Injuries/Fatal')
        dialog1.s.ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
        dialog1.s.ax.grid(True)
        dialog1.s.draw()
        self.dialogs.append(dialog1)
        dialog1.show()

    def EDA3(self):
        #::----------------------------------------------------------
        # This function creates an instance of the Crash_Graphsclass
        #::----------------------------------------------------------
        dialog = Crash_Graphs()
        self.dialogs.append(dialog)
        dialog.show()

    def LG(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = Logit()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def XGBoost(self):
        dialog = Boosted()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def crash_data():
    #::--------------------------------------------------
    # Loads the crash.csv file created from readdata.py
    #
    #::--------------------------------------------------
    global X
    global y
    global features_list
    global columns_list
    global class_names
    global fatal_crash
    global crash
    global top_10
    global crash_model
    crash = pd.read_csv('crash.csv')
    fatal_crash =crash[crash.FATALMAJORINJURIES.eq(1.0)]
    fatal_crash.dropna(inplace=True)
    top_10 = fatal_crash.head()
    columns_list = ['PERSONID', 'PERSONTYPE', 'AGE', 'FATAL', 'MAJORINJURY', 'MINORINJURY', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING', 'FATALMAJORINJURIES']
    X=pd.Series(fatal_crash['AGE'])
    X.dropna(inplace=True)
    y = crash["FATALMAJORINJURIES"]
    crash_model = pd.read_csv('crash_model.csv')
    features_list = ['PERSONTYPE', 'AGE',  'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']
    class_names = ['FATAL', 'NOT FATAL']


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    crash_data()
    main()