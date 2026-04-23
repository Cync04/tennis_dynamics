import pandas as pd
import matplotlib.pyplot as plt

def read_lines_pandas(filename, start_line, num_rows):
   
    data = pd.read_csv(
        filename, 
        skiprows=start_line - 1, 
        nrows=num_rows, 
        usecols= ["PointWinner","PointServer", "ServeWidth"]
    )
    return data



df2024 = read_lines_pandas('2024-wimbledon-points.csv', 1, 48150)
df2023 = read_lines_pandas('2023-wimbledon-points.csv', 1, 48677)
df2022 = read_lines_pandas('2022-wimbledon-points.csv', 1, 46362)

df = pd.concat([df2024, df2023, df2022], ignore_index=True)

c = 0
c2 = 0
bc = 0
bc2 = 0
b = 0
b2 = 2
bw = 0
bw2 = 0
w = 0
w2 = 0
count1 = 0
count2 = 0

for point in range(len(df)):
    location = df.loc[point, "ServeWidth"]
    if location == "":
        continue
    count1 += 1
    if df.loc[point, "PointWinner"] == df.loc[point, "PointServer"]:
        
        if location == "C":
            c += 1
        elif location == "BC":
            bc += 1
        elif location == "B":
            b += 1
        elif location == "BW":
            bw += 1
        elif location == "W":
            w += 1
    else:
        
        if location == "C":
            c2 += 1
        elif location == "BC":
            bc2 += 1
        elif location == "B":
            b2 += 1
        elif location == "BW":
            bw2 += 1
        elif location == "W":
            w2 += 1

positions = ["W", "BW", "B", "BC", "C"]
heights = [(w/(w+w2))*100, (bw/(bw+bw2))*100, (b/(b+b2))*100, (bc/(bc+bc2))*100, (c/(c+c2))*100]

plt.ylim(40, 100)
plt.bar(positions, heights)
plt.ylabel("Percent Chance of Winning")
plt.xlabel("Serve Location")
plt.title("Actual Chance of Winning")
plt.show()