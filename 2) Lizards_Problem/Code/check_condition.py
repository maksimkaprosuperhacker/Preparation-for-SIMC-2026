import pandas as pd

df = pd.read_csv("lizards_data.csv", sep=',')

df = df.sort_values(by = 'Species_num', ascending = True).reset_index(drop = True)
total=0
GROUP=7

def condition(row):
    return row["VSN"] > 26.10 and row["SDLr"] < 21.18 and row["SCGr"] > 9.18


for i in range(len(df)):
    row = df.loc[i]
    if row["Species_num"]==GROUP:
        total+=1

true_predict=0
false_predict=0
for i in range(len(df)):
    row = df.loc[i]
    if condition(row):
        if row["Species_num"] == GROUP:
            true_predict += 1
        else:
            false_predict+=1
print(f"Total: {total}")
print(f"True: {true_predict}")
print(f"False: {false_predict}")
print(f"Acc: {1 - (total - true_predict + false_predict) / len(df)}")

