import pandas as pd
import numpy as np
import itertools


TARGET_SPECIES = [1]


BACKGROUND_SPECIES = [1, 2,3,4,5,6,7,8]


STEPS = 1000

# final answers arr
super_results = []

df = pd.read_csv("lizards_data.csv", sep = ',')


PARAMETERS = ["MBS", "VSN", "CSN", "GSN", "FPNr", "SDLr", "SCSr", "SCGr", "SMr", "MTr", "PA", "PTMr", "aNDSr", "SVL",
              "TRL", "HL", "PL", "ESD", "HW", "HH", "MO", "FFL", "HFL"]

# if we need filter
relevant_mask = df["Species_num"].isin(TARGET_SPECIES + BACKGROUND_SPECIES)
df_clean = df[relevant_mask].copy()


# our target group -> True
# others -> False
arr_of_true = df_clean["Species_num"].isin(TARGET_SPECIES).to_numpy()

print(f"{TARGET_SPECIES} vs {BACKGROUND_SPECIES}")
print(f"Total: {len(arr_of_true)}")
print(f"Total in {TARGET_SPECIES} : {np.sum(arr_of_true)}")
print(f"Others : {len(arr_of_true) - np.sum(arr_of_true)}")



# all logic variants
# 1 -> ">"
# -1 -> "<"
logic_configs = [
    ("p1 > t1 AND p2 > t2", 1, '&', 1),
    ("p1 < t1 AND p2 < t2", -1, '&', -1),
    ("p1 < t1 AND p2 > t2", -1, '&', 1),
    ("p1 > t1 AND p2 < t2", 1, '&', -1),
    ("p1 > t1 OR p2 > t2", 1, '|', 1),
    ("p1 < t1 OR p2 < t2", -1, '|', -1),
    ("p1 < t1 OR p2 > t2", -1, '|', 1),
    ("p1 > t1 OR p2 < t2", 1, '|', -1),
]

# take 2 parameters from permutations
for p1_name, p2_name in itertools.permutations(PARAMETERS, 2):

    # 2 arrays of 2 parameters from main arr
    v1 = df_clean[p1_name].to_numpy()
    v2 = df_clean[p2_name].to_numpy()

    # range of values of 2 parameters
    t1_vals = np.linspace(v1.min(), v1.max(), STEPS)
    t2_vals = np.linspace(v2.min(), v2.max(), STEPS)

    # Without cycles pretty fast
    # how many p1's > t1_vals
    m1_g = v1[:, None] > t1_vals[None, :]
    # how many p1's < t1_vals
    m1_l = v1[:, None] < t1_vals[None, :]


    # how many p2's > t2_vals
    m2_g = v2[:, None] > t2_vals[None, :]
    # how many p2's < t2_vals
    m2_l = v2[:, None] < t2_vals[None, :]

    #column of arr_of_true
    arr_of_true_col = arr_of_true[:, None]

    # check all conditions
    for label, op1, logic, op2 in logic_configs:
        mat1 = m1_g if op1 == 1 else m1_l
        mat2 = m2_g if op2 == 1 else m2_l

        best_local_acc = -1
        best_t1 = 0
        best_t2 = 0

        for i in range(STEPS):
            col1 = mat1[:, i:i + 1]

            if logic == '&':
                preds = col1 & mat2
            else:
                preds = col1 | mat2  #

            accuracies = (preds == arr_of_true_col).mean(axis = 0)

            max_idx = np.argmax(accuracies)
            current_max = accuracies[max_idx]

            if current_max > best_local_acc:
                best_local_acc = current_max
                best_t1 = t1_vals[i]
                best_t2 = t2_vals[max_idx]

        super_results.append({
            "Params":[p1_name, p2_name],
            "Logic":label,
            "T1":best_t1,
            "T2":best_t2,
            "Accuracy":best_local_acc
        })


super_results.sort(key = lambda x:x["Accuracy"], reverse = True)



if super_results:
    top = super_results[0]
    print(f"Parameter {top['Params'][0]} and {top['Params'][1]}")

    formula = top['Logic'].replace('p1', top['Params'][0]).replace('p2', top['Params'][1])
    formula = formula.replace('t1', f"{top['T1']:.2f}").replace('t2', f"{top['T2']:.2f}")

    print(f"Formula:   {formula}")
    print(f"Accuracy:  {top['Accuracy']:.2%}")


    for i, res in enumerate(super_results[:5]):
        rule = res['Logic'].replace('p1', res['Params'][0]).replace('p2', res['Params'][1])
        rule = rule.replace('t1', f"{res['T1']:.2f}").replace('t2', f"{res['T2']:.2f}")
        print(f"{i + 1}. {rule} | Acc: {res['Accuracy']:.2%}")
else:
    print("ERROR")