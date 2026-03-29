import pandas as pd

def read_lines_pandas(filename, start_line, num_rows):
   
    data = pd.read_csv(
        filename, 
        skiprows=start_line - 1, 
        nrows=num_rows, 
        usecols= ["PointWinner","PointServer", "ServeWidth"]
    )
    return data



df = read_lines_pandas('2024-wimbledon-points.csv', 1, 48000)
print(df)

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
    if df.loc[point, "ServeWidth"] == "":
        continue
    count1 += 1
    if df.loc[point, "PointWinner"] == df.loc[point, "PointServer"]:
        
        if df.loc[point, "ServeWidth"] == "C":
            c += 1
        elif df.loc[point, "ServeWidth"] == "BC":
            bc += 1
        elif df.loc[point, "ServeWidth"] == "B":
            b += 1
        elif df.loc[point, "ServeWidth"] == "BW":
            bw += 1
        elif df.loc[point, "ServeWidth"] == "W":
            w += 1
    else:
        
        if df.loc[point, "ServeWidth"] == "C":
            c2 += 1
        elif df.loc[point, "ServeWidth"] == "BC":
            bc2 += 1
        elif df.loc[point, "ServeWidth"] == "B":
            b2 += 1
        elif df.loc[point, "ServeWidth"] == "BW":
            bw2 += 1
        elif df.loc[point, "ServeWidth"] == "W":
            w2 += 1

print(f"Percent of points served to center: {((c+c2)/count1)*100:.1f}")
print(f"Percent of points served to body-center: {((bc+bc2)/count1)*100:.1f}")
print(f"Percent of points served to body: {((b+b2)/count1)*100:.1f}")
print(f"Percent of points served to body-wide: {((bw+bw2)/count1)*100:.1f}")
print(f"Percent of points served to wide: {((w+w2)/count1)*100:.1f}\n")

print(f"Percent of points won at center: {(c/(c+c2))*100:.1f}")
print(f"Percent of points won at body-center: {(bc/(bc+bc2))*100:.1f}")
print(f"Percent of points won at body: {(b/(b+b2))*100:.1f}")
print(f"Percent of points won at body-wide: {(bw/(bw+bw2))*100:.1f}")
print(f"Percent of points won at wide: {(w/(w+w2))*100:.1f}")