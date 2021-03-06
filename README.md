# Final-Project-Group3
Group project for DATS 6103 Data Mining, Spring 2021 with Prof. Jafari in George Washington University (Washington DC).
Included is the work on the class project with fellow teammates Arianna Dunham (adunham@gwmail.gwu.edu) and RyeAnne Ricker (ryeannericker@gwu.edu)

Project is based on OpenData DC's Crash_Details_Table.csv which has been downloaded to the \code directory.
District Department of Transportation, Metropolitan Police Department, Crashes Details Table, Open Data DC, (District of Columbia): Vision Zero Data Planning Work Group, 2020. Accessed on: Mar. 14, 2020. [online]. 
Available: https://opendata.dc.gov/datasets/crash-details-table

Order in which to run the codes:
1. readdata.py - reads the data csv file and creates crash dataframe, adds a target "FATALMAJORINJURIES" = 1 if "FATAL" or "MAJORINJURY" column is "Y"
2. eda.py - performs the exploratory data analysis of dataset 
3. stats.py - statistical analysis to determine whether features are independent of a fatality/major injury & analyze age differences between target group
4. preprocessing.py - clean data, fill missing data, normalization, creation of feature matrix
5. BoostedDT.py - Extreme Gradient Boosted Decision Tree model
6. NaiveBayes.py - Naive Bayes Classifier model
7. randomforest.py - Random Forest Classifier model
8. Logit.py - Logistic Regression Classifier model
9. VotingClassifier.py - Voting Classifier model
10. Main.py - Crash data desktop application/GUI using PyQt5

When running eda.py, stats.py, or preprocessing.py, the data will be imported from readdata, the file that downloads the data from the internet and does some preliminary cleaning. 

When running any of the models, the feature matrix with target will be imported from the preprocessing script. The models are all cross validated. If the user is concerned about run time, however, the lines for cross validation may be commented out. The lines to be commented out are clearly labelled in the scripts. 



