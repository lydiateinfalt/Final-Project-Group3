# This portion determines whether features are independent of crash having a fatality/major injury
# Chi squared tests are performed on each categorical variables to determine independence

# RR - 30 lines, 15 copied and modified, 15 myself
# Ward
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["WARD"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Ward are:',test_results)
# Age
#crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["AGE"],test= "chi-square",expected_freqs= True,prop= "cell")
#print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Age are:',test_results)
# Total Vehicles
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TOTAL_VEHICLES"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Total Vehicles are:',test_results)
# Total Bicycles
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TOTAL_BICYCLES"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Total Bicycles are:',test_results)
# Total Pedestrians
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TOTAL_PEDESTRIANS"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Total Pedestrians are:',test_results)
# Drivers Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["DRIVERSIMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired Drivers are:',test_results)
# Pedestrians Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["PEDESTRIANSIMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired Pedestrians are:',test_results)
# Bicyclists Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["BICYCLISTSIMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired Bicyclists are:',test_results)
# Intersection
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["OFFINTERSECTION"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Intersection are:',test_results)
# Vehicle Type
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["INVEHICLETYPE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Vehicle Type are:',test_results)
# Ticket Issued
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["TICKETISSUED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Tickets Issued are:',test_results)
# Lice Plate State
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["LICENSEPLATESTATE"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and License Plate State are:',test_results)
# Impaired
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["IMPAIRED"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Impaired are:',test_results)
# Speeding
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["SPEEDING"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Speeding are:',test_results)
# Year
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["YEAR"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Year are:',test_results)
# Month
crosstab, test_results, expected = rp.crosstab(crash["FATALMAJORINJURIES"], crash["MONTH"],test= "chi-square",expected_freqs= True,prop= "cell")
print('The Chi-Squared test results for the relationship between Fatal/Major Injuries and Month are:',test_results)
