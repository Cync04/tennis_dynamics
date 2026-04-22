import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

def read_lines_pandas(filename, start_line, num_rows):
   
    data = pd.read_csv(
        filename, 
        skiprows=start_line - 1, 
        nrows=num_rows, 
        usecols= ["PointWinner","PointServer","Speed_MPH"]
    )
    return data



df2024 = read_lines_pandas('2024-wimbledon-points.csv', 1, 48150)
df2023 = read_lines_pandas('2023-wimbledon-points.csv', 1, 48677)
df2022 = read_lines_pandas('2022-wimbledon-points.csv', 1, 46362)

df = pd.concat([df2024, df2023, df2022], ignore_index=True)

winData = dict()
for point in range(len(df)):
    speed = df.loc[point, "Speed_MPH"]
    if speed == 0:
        continue
    if df.loc[point, "PointWinner"] == df.loc[point, "PointServer"]:
        if winData.get(speed) == None:
            winData[speed] = [1, 0]
        else:
            winData[speed][0] += 1
    else:
        if winData.get(speed) == None:
            winData[speed] = [0, 1]
        else:
            winData[speed][1] += 1

x = np.array([])
y = np.array([])
sorted_dict = dict(sorted(winData.items()))
for speedKey in sorted_dict.keys():
    x = np.append(x, speedKey)
    wins = sorted_dict[speedKey][0]
    loss = sorted_dict[speedKey][1]
    y = np.append(y, wins/(loss + wins)) 
    
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
dataSet, trash = optimize.curve_fit(quadratic, x, y)
a, b, c = dataSet[:]

# Creating graph
plt.plot(x, y, label="Data")
plt.plot(x, a*x**2 + b*x + c, label="Model")
plt.title("Probability of Point Win Based on Serve Speed")
plt.xlabel("Serve Speed (MPH)")
plt.ylabel("Probability of Win")
plt.legend(loc=2)
plt.grid()
plt.show()