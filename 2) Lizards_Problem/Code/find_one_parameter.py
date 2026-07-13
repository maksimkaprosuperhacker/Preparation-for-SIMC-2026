import pandas as pd
import numpy as np


df = pd.read_csv("lizards_data.csv", sep = ',')


GROUP=7

PARAMETERS = ["MBS", "VSN", "CSN", "GSN", "FPNr", "SDLr", "SCSr", "SCGr", "SMr", "MTr", "PA", "PTMr", "aNDSr", "SVL",
              "TRL", "HL", "PL", "ESD", "HW", "HH", "MO", "FFL", "HFL"]
super_results = []


actual_species = (df["Species_num"] == GROUP).to_numpy()

for param in PARAMETERS:
    values = df[param].to_numpy()

    min_v = np.min(values)
    max_v = np.max(values)
    thresholds = np.linspace(min_v, max_v, 1000)

    best_acc = -1
    best_thresh = 0

    for thresh in thresholds:

        predict_is_1 = values < thresh

        correct = (predict_is_1 == actual_species)
        acc = np.mean(correct)


        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    super_results.append([param, best_thresh, best_acc])


super_results.sort(key = lambda x:x[2], reverse = True)

print("Best Parameter:", super_results[0][0])
print("Threshold:", float(super_results[0][1]))
print("Accuracy:", float(super_results[0][2]))

for i in range(3):
    print(f"{i + 1}. {super_results[i][0]}: {super_results[i][2]:.2%}")