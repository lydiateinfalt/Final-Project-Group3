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
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
#from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

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
        self.Title = 'DC Fatal/Major Injuries Histogram'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

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

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        DataButton = QAction(QIcon(),'Sample ', self)
        DataButton.setStatusTip('Sample data')
        DataButton.triggered.connect(self.Data1)
        DataMenu.addAction(DataButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        EDA1Button = QAction(QIcon(),'Histogram', self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are three models
        #       SVM
        #       XGBoost
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button =  QAction(QIcon(), 'SVM', self)
        MLModel1Button.setStatusTip('SVM')
        MLModel1Button.triggered.connect(self.SVM)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)

        self.dialogs = list()

    # Reference: https://pythonspot.com/pyqt5-table/
    def Data1(self):
        dialog = DataTable(self)
        #dialog.t.setWindowTitle('DC Crash Data')
        # Set number of rows
        dialog.t.setRowCount(top_10.shape[0])
        num_cols = len(features_list)
        dialog.t.setColumnCount(num_cols)

        # Set column headings
        for i in range(0,num_cols):
            dialog.t.setItem(0, i, QTableWidgetItem(features_list[i]))

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
        # This function creates an instance of HappinessGraphs class
        # This class creates a graph using the features in the dataset
        # happiness vrs the score of happiness
        #::---------------------------------------------------------
        dialog = ()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = ()
        self.dialogs.append(dialog)
        dialog.show()

    def SVM(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = ()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = ()
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
    global class_names
    global top_10
    crash = pd.read_csv('crash.csv')
    fatal_crash =crash[crash.FATALMAJORINJURIES.eq(1.0)]
    fatal_crash.dropna(inplace=True)
    X=pd.Series(fatal_crash['AGE'])
    X.dropna(inplace=True)
    top_10=fatal_crash.head()
    y = crash["FATALMAJORINJURIES"]
    features_list = ['PERSONID', 'PERSONTYPE', 'AGE', 'FATAL', 'MAJORINJURY', 'MINORINJURY', 'INVEHICLETYPE', 'TICKETISSUED', 'LICENSEPLATESTATE', 'IMPAIRED', 'SPEEDING']
    class_names = ['FATAL', 'NOT FATAL']


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    crash_data()
    main()