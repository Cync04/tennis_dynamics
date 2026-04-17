import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

def read_lines_pandas(filename, start_line, num_rows):
   
    data = pd.read_csv(
        filename, 
        skiprows=start_line - 1, 
        nrows=num_rows, 
        usecols= ["ElapsedTime", "Speed_MPH"]
    )
    return data

# Reading point data
df2024 = read_lines_pandas('2024-wimbledon-points.csv', 1, 48150)
df2023 = read_lines_pandas('2023-wimbledon-points.csv', 1, 48677)
df2022 = read_lines_pandas('2022-wimbledon-points.csv', 1, 46362)

df = pd.concat([df2024, df2023, df2022], ignore_index=True)

# Cleaning data and inputting into a dictionary with key of the minute and value of a list of the serve speeds
minuteData = dict()
for point in range(len(df)):
    if df.loc[point, "Speed_MPH"] == 0:
        continue
    time = df.loc[point, "ElapsedTime"]
    if int(time[0]) > 5 or time[1] != ":":
        continue
    time = ((int(time[0])*60) + int(time[2:4]))
    if minuteData.get(time) == None:
        minuteData[time] = [df.loc[point, "Speed_MPH"]]
    else:
        minuteData[time].append(df.loc[point, "Speed_MPH"])

# Switching from dictionary to lists for x and y values
times = np.array([])
speedAvg = np.array([])
sorted_dict = dict(sorted(minuteData.items()))
for minuteKey in sorted_dict.keys():
    times = np.append(times, minuteKey)
    speedAvg = np.append(speedAvg, (sum(sorted_dict[minuteKey])/len(sorted_dict[minuteKey]))) # Averaging speed within the minute

# Creating model
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
dataSet, trash = optimize.curve_fit(quadratic, times, speedAvg)
a, b, c = dataSet[:]

# Creating graph
plt.plot(times, speedAvg, label="Data")
plt.plot(times, a*times**2 + b*times + c, label="Model")
plt.title("Speed Averages Compared to Match Length")
plt.xlabel("Match Length (min)")
plt.ylabel("Average Speed (MPH)")
plt.legend(loc=2)
plt.grid()
plt.show()


