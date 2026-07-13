import pandas as pd

ANS=12

df = pd.read_csv("lizards_data.csv", sep=',')

df = df.sort_values(by = 'Species_num', ascending = True).reset_index(drop = True)

total=0

for i in range(len(df)):
    row = df.loc[i]
    if row["Species_num"]==5:
        total+=1

true_predict=0
false_predict=0
for i in range(len(df)):
    row = df.loc[i]
    if row["FPNr"]<ANS:
        if row["Species_num"] == 5:
            true_predict += 1
        else:
            false_predict+=1
print(total)
print(true_predict)
print(false_predict)
print(total-true_predict+false_predict)

