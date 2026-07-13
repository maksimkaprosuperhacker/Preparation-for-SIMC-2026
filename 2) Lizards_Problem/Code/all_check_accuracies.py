import pandas as pd
import numpy as np


df = pd.read_csv("lizards_data.csv", sep = ',')


rules = [
    (1, lambda r:r["GSN"] < 27.63 and r["FPNr"] > 16.47 and r["MTr"] > 2.21),
    (2, lambda r:r["MBS"] > 45.14 and r["PTMr"] > 3.08 and r["ESD"] < 7.05),
    (3, lambda r:r["PA"] > 1.00 and r["aNDSr"] > 2.52 and r["TRL"] < 33.67),
    (4, lambda r:r["GSN"] > 26.13 and r["SCGr"] < 8.03),
    (5, lambda r:r["MBS"] < 41.09 and r["FPNr"] < 11.18),
    (6, lambda r:r["VSN"] > 25.06 and r["SDLr"] < 21.18 and r["SCGr"] < 9.18),
    (7, lambda r:r["VSN"] > 26.10 and r["SDLr"] < 21.18 and r["SCGr"] > 9.18),
    (8, lambda r:r["MBS"] < 47.27 and r["ESD"] < 7.30 and r["FFL"] > 19.55)
]

accuracies = []
n_samples = len(df)

print(f"{'Group ID':<10} | {'Total':<6} | {'True':<10} | {'False':<10} | {'Accuracy':<10}")


for group_id, condition in rules:
    total_actual = (df["Species_num"] == group_id).sum()
    true_predict = 0
    false_predict = 0

    for _, row in df.iterrows():

        is_predicted = condition(row)


        is_actual = (row["Species_num"] == group_id)

        if is_predicted:
            if is_actual:
                true_predict += 1
            else:
                false_predict += 1


    errors = (total_actual - true_predict) + false_predict
    accuracy = 1 - (errors / n_samples)
    accuracies.append(accuracy)

    print(f"{group_id:<10} | {total_actual:<6} | {true_predict:<10} | {false_predict:<10} | {accuracy:.2%}")

mean_accuracy = np.mean(accuracies)

print(f"OVERALL MEAN ACCURACY: {mean_accuracy:.2%}")