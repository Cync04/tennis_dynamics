import pandas as pd

def read_lines_pandas(filename, start_line, num_rows):
   
    data = pd.read_csv(
        filename, 
        skiprows=start_line - 1, 
        nrows=num_rows, 
        usecols= ["PointWinner","PointServer","Speed_MPH", "ServeIndicator"]
    )
    return data



df = read_lines_pandas('2024-wimbledon-points.csv', 1, 48000)
print(df)

win1 = []
loss1 = []
win2 = []
loss2 = []

for point in range(len(df)):
    if df.loc[point, "Speed_MPH"] == 0:
        continue
    if df.loc[point, "PointWinner"] == df.loc[point, "PointServer"]:
        if df.loc[point, "ServeIndicator"] == 1:
            win1.append(df.loc[point, "Speed_MPH"])
        else:
            win2.append(df.loc[point, "Speed_MPH"])
    else:
        if df.loc[point, "ServeIndicator"] == 1:
            loss1.append(df.loc[point, "Speed_MPH"])
        else:
            loss2.append(df.loc[point, "Speed_MPH"])
print("\nWon the point on first serve:", sum(win1)/len(win1))
print("Won the point on second serve:", sum(win2)/len(win2))
print("\nLost the point on first serve:", sum(loss1)/len(loss1))
print("Lost the point on second serve:", sum(loss2)/len(loss2))