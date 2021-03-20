import data_cleanup
import pandas as pd
import matplotlib.pyplot as plt

data = data_cleanup.crash

fatalcrashes = data[( data['FATAL_BICYCLIST'] > 0) | ( data['FATAL_DRIVER'] > 0) | ( data['FATAL_PEDESTRIAN'] > 0)]
majorcrashes = data[(data['MAJORINJURIES_BICYCLIST'] > 0) | (data['MAJORINJURIES_DRIVER'] > 0) | (data['MAJORINJURIES_PEDESTRIAN'] > 0)]

f = pd.DataFrame(fatalcrashes.groupby(['YEAR']).size())
m = pd.DataFrame(majorcrashes.groupby(['YEAR']).size())
t= m + f

# plot number of accidents per year
fig, ax = plt.subplots()
fig.set_size_inches(12,10)
ax.bar(m.index, m[0], color = "blue")
ax.bar(f.index, f[0], color = "red")

labels = ("Major Injuries", "Fatalities")
ax.set_xlabel('Year')
ax.set_title('Number of Crashes in DC Resulting in Fatalities/Major Injuries')
ax.legend(labels)
plt.show()


# No fatality, no major crashes
no_fatal_crashes = data[(data['FATAL_BICYCLIST'] == 0) | (data['FATAL_DRIVER'] == 0) | (data['FATAL_PEDESTRIAN'] == 0)]
no_major_crashes = data[(data['MAJORINJURIES_BICYCLIST'] == 0 ) | (data['MAJORINJURIES_DRIVER'] == 0) | (data['MAJORINJURIES_PEDESTRIAN'] == 0)]

f0 = pd.DataFrame(no_fatal_crashes.groupby(['YEAR']).size())
m0 = pd.DataFrame(no_major_crashes.groupby(['YEAR']).size())
t0 = m0 + f0
fig, ax = plt.subplots()
fig.set_size_inches(16,12)
ax.bar(t0.index, f0[0], color = "green")
ax.set_xlabel('Year')
ax.set_title('Number of Crashes Without Fatalities')
plt.show()
