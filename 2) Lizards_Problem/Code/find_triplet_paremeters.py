import pandas as pd
import numpy as np
import itertools
from time import time
from numba import njit, prange


TARGET_SPECIES = [8]
BACKGROUND_SPECIES = [1, 2, 3, 4, 5, 6, 7,8]
STEPS = 10



"""
Load Data → Select Target Species → Compress Data into Bits (for speed) →
Prepare All Thresholds → Run Parallel Search (using all CPU cores) →
Test Combinations of 3 Parameters → Compare Using Fast Bitwise Logic →
Count Correct Matches → Save Best Score → Print Top Results

"""




df = pd.read_csv("lizards_data.csv", sep = ',')


PARAMETERS = ["MBS", "VSN", "CSN", "GSN", "FPNr", "SDLr", "SCSr", "SCGr", "SMr", "MTr", "PA", "PTMr", "aNDSr", "SVL",
              "TRL", "HL", "PL", "ESD", "HW", "HH", "MO", "FFL", "HFL"]

df_clean = df[df["Species_num"].isin(TARGET_SPECIES + BACKGROUND_SPECIES)].copy()

# our target group -> True
# others -> False
arr_of_true = df_clean["Species_num"].isin(TARGET_SPECIES).to_numpy().astype(bool) #


arr_of_true_packed = np.packbits(arr_of_true)
real_n_samples = len(arr_of_true)
n_bytes = len(arr_of_true_packed)

n_params = len(PARAMETERS)
masks_packed = np.zeros((n_params, STEPS, n_bytes), dtype = np.uint8)
thresholds_all = np.zeros((n_params, STEPS), dtype = np.float32)

for i, param in enumerate(PARAMETERS):
    vals = df_clean[param].to_numpy()
    th = np.linspace(vals.min(), vals.max(), STEPS)
    thresholds_all[i, :] = th
    bool_matrix = (vals[:, None] > th[None, :])
    masks_packed[i, :, :] = np.packbits(bool_matrix.T, axis = 1)


POP_COUNT_TABLE = np.array([bin(x).count('1') for x in range(256)], dtype = np.uint8) # table where arr[i] = number of 1s in binary representation i



@njit(fastmath = True, parallel = True)
def fast_search(masks, y_packed, pop_table, steps, n_params, n_bytes, real_n_samples):
    results = np.zeros((n_params, n_params, n_params, 6), dtype = np.float32)

    remainder = real_n_samples % 8
    last_byte_mask = np.uint8(255)
    if remainder > 0:
        last_byte_mask = np.uint8(0xFF << (8 - remainder))




    # check all variants "p1", "p2", and "p3
    for p1 in prange(n_params): # fast cycle from numba
        for p2 in range(p1 + 1, n_params):
            for p3 in range(p2 + 1, n_params):

                m1_pack = masks[p1]
                m2_pack = masks[p2]
                m3_pack = masks[p3]

                best_acc = -1.0
                curr_res = np.zeros(6, dtype = np.float32)


                # iterate over all threshold combinations
                for t1 in range(steps):
                    r1 = m1_pack[t1]
                    for t2 in range(steps):
                        r2 = m2_pack[t2]
                        for t3 in range(steps):
                            r3 = m3_pack[t3]



                            # check all variants "<" and ">"
                            for s1 in range(2):
                                for s2 in range(2):
                                    for s3 in range(2):
                                        sign_code = (s1 << 2) | (s2 << 1) | s3

                                        hits_and = 0
                                        hits_or = 0

                                        for b in range(n_bytes):
                                            v1 = r1[b] if s1 else ~r1[b]
                                            v2 = r2[b] if s2 else ~r2[b]
                                            v3 = r3[b] if s3 else ~r3[b]

                                            yt = y_packed[b]


                                            mask_p = last_byte_mask if b == n_bytes - 1 else np.uint8(255)

                                            #check how many correct
                                            match_and = (~((v1 & v2 & v3) ^ yt)) & mask_p
                                            hits_and += pop_table[match_and]

                                            match_or = (~((v1 | v2 | v3) ^ yt)) & mask_p
                                            hits_or += pop_table[match_or]


                                        # save result in curr_res[] and then in results[]
                                        acc_and = hits_and / real_n_samples
                                        if acc_and > best_acc:
                                            best_acc = acc_and
                                            curr_res[0] = acc_and
                                            curr_res[1] = t1
                                            curr_res[2] = t2
                                            curr_res[3] = t3
                                            curr_res[4] = sign_code
                                            curr_res[5] = 0.0

                                        acc_or = hits_or / real_n_samples
                                        if acc_or > best_acc:
                                            best_acc = acc_or
                                            curr_res[0] = acc_or
                                            curr_res[1] = t1
                                            curr_res[2] = t2
                                            curr_res[3] = t3
                                            curr_res[4] = sign_code
                                            curr_res[5] = 1.0

                results[p1, p2, p3] = curr_res
    return results



start_time = time()

res_cube = fast_search(masks_packed, arr_of_true_packed, POP_COUNT_TABLE, STEPS, n_params, n_bytes, real_n_samples)

print(f"DONE in {time() - start_time:.4f} seconds")


# RESULTS PARSING

final_results = []
for p1, p2, p3 in itertools.combinations(range(n_params), 3):
    r = res_cube[p1, p2, p3]
    acc = r[0]

    if acc > 0.5:
        t1, t2, t3 = int(r[1]), int(r[2]), int(r[3])
        signs = int(r[4])
        logic = int(r[5])

        s1 = ">" if (signs >> 2) & 1 else "<"
        s2 = ">" if (signs >> 1) & 1 else "<"
        s3 = ">" if (signs >> 0) & 1 else "<"
        l_str = "AND" if logic == 0 else "OR"

        logic_str = (f"{PARAMETERS[p1]} {s1} {thresholds_all[p1, t1]:.2f} {l_str} "
                     f"{PARAMETERS[p2]} {s2} {thresholds_all[p2, t2]:.2f} {l_str} "
                     f"{PARAMETERS[p3]} {s3} {thresholds_all[p3, t3]:.2f}")

        final_results.append({"Logic":logic_str, "Accuracy":acc})

final_results.sort(key = lambda x:x["Accuracy"], reverse = True)

seen = set()
count = 0
for res in final_results:
    if count >= 10: break
    sig = res['Logic']
    if sig not in seen:
        print(f"{count + 1}. {res['Logic']} | Acc: {res['Accuracy']:.2%}")
        seen.add(sig)
        count += 1