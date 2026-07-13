import pandas as pd

df = pd.read_csv("lizards_data.csv", sep=',')

df = df.sort_values(by = 'Species_num', ascending = True).reset_index(drop = True)
goal=5
num_of_group=8
results = {}
parameters_list=["MBS","VSN","CSN","GSN","FPNr","SDLr","SCSr","SCGr","SMr","MTr","PA","PTMr","aNDSr","SVL","TRL","HL","PL","ESD","HW","HH","MO","FFL","HFL"]
for parameter in parameters_list:
    means={}
    prev = df.loc[0, "Species_num"]
    s=0
    count=0
    for i in range(len(df)):
        row = df.loc[i]
        if row["Species_num"] == prev:
            s+=row[parameter]
            count+=1
        else:
            means[str(prev)]=float(s/count)
            prev=row["Species_num"]
            s=row[parameter]
            count=1
    means[str(prev)] = float(s / count)
    main=means[str(goal)]
    sub_res=0
    for i in means:
        if i != str(goal):
            sub_res += means[i] - main
    sub_res=sub_res/(num_of_group-1)
    results[parameter]=sub_res

for k, v in sorted(results.items(), key=lambda x: x[1], reverse = True):
    print(f"{k} : {v}")