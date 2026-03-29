import pandas as pd

def read_lines_pandas(filename, start_line, num_rows):
   
    data = pd.read_csv(
        filename, 
        skiprows=start_line - 1, 
        nrows=num_rows, 
        usecols= ["P2Ace","P1Ace", "Speed_MPH"]
    )
    return data

df = read_lines_pandas('2024-wimbledon-points.csv', 1, 48000)
aceSpeed = []
notAceSpeed = []
for point in range(len(df)):
    if df.loc[point, "Speed_MPH"] == 0:
        continue
    if df.loc[point, "P1Ace"] == 1 or df.loc[point, "P2Ace"] == 1:
        aceSpeed.append(df.loc[point, "Speed_MPH"])
    else:
       notAceSpeed.append(df.loc[point, "Speed_MPH"]) 

print(f"Average serve speed for aces: {(sum(aceSpeed)/len(aceSpeed)):.1f}")
print(f"Average serve speed for non-aces: {(sum(notAceSpeed)/len(notAceSpeed)):.1f}")