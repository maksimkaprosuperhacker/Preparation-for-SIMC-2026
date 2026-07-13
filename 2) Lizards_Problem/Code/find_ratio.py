import pandas as pd
import numpy as np

df = pd.read_csv("lizards_data.csv", sep = ',')

GROUP=8
STEPS=100
PARAMETERS = ["MBS", "VSN", "CSN", "GSN", "FPNr", "SDLr", "SCSr", "SCGr", "SMr", "MTr", "PA", "PTMr", "aNDSr", "SVL",
              "TRL", "HL", "PL", "ESD", "HW", "HH", "MO", "FFL", "HFL"]

super_results = []

actual_species = (df["Species_num"] == GROUP).to_numpy()

for p1 in PARAMETERS:
    for p2 in PARAMETERS:
        if p1 == p2:
            continue

        # all rations of p1 and p2
        ratios = df[p1].to_numpy() / df[p2].to_numpy()



        min_v = np.min(ratios)
        max_v = np.max(ratios)

        thresholds = np.linspace(min_v, max_v, STEPS)

        best_acc = -1
        best_thresh = 0

        for thresh in thresholds:
            predict_is_correct = ratios > thresh

            correct = (predict_is_correct == actual_species)
            acc = np.mean(correct)


            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        super_results.append([[p1, p2], best_thresh, best_acc])

super_results.sort(key = lambda x:x[2], reverse = True)


print(f"UP: {super_results[0][0][0]}")
print(f"DOWN: {super_results[0][0][1]}")
print(f"VALUE: {super_results[0][1]:.4f}")
print(f"ACC: {super_results[0][2]:.2%}")


for i in range(3):
    p1, p2 = super_results[i][0]
    acc = super_results[i][2]
    print(f"{i + 1}. {p1}/{p2}: {acc:.2%}")