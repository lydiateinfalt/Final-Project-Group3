# Final-Project-Group3
Group project for DATS 6103 Data Mining, Spring 2021 with Prof. Jafari in George Washington University (Washington DC).
Included is the work on the class project with fellow teammates Arianna Dunham (adunham@gwmail.gwu.edu) and RyeAnne Ricker (ryeannericker@gwu.edu)

Project is based on OpenData DC's Crash_Details_Table.csv which has been downloaded to the \code directory.
District Department of Transportation, Metropolitan Police Department, Crashes Details Table, Open Data DC, (District of Columbia): Vision Zero Data Planning Work Group, 2020. Accessed on: Mar. 14, 2020. [online]. 
Available: https://opendata.dc.gov/datasets/crash-details-table

1. readdata.py - reads the data csv file and creates crash dataframe, adds a target "FATALMAJORINJURIES" = 1 if "FATAL" or "MAJORINJURY" column is "Y"
2. eda.py - exploratory data analysis dataset 
3. stats.py - statistical analysis to determine whether features are independent of crash having a fatality/major injury
4. preprocessing.py - Clean up invalid data, fill missing data and filter out certain conditions



