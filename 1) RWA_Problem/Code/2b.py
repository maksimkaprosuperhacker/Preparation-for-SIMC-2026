import pandas as pd
import numpy as np


def solve_2b():
    df = pd.read_csv("2b.csv", sep = ';')
    df = df.iloc[:, :6]
    df.columns = ["asset", "liab", "m_d", "m_p", "ir", "gn"]

    df['asset'] = df['asset'].astype(float)
    df['liab'] = df['liab'].astype(float)
    df['ir'] = df['ir'].astype(float)
    df['m_p_int'] = pd.to_numeric(df['m_p'], errors = 'coerce').fillna(0).astype(int)
    df['val'] = df['asset'] - df['liab']

    df = df.sort_values(by = ['ir', 'm_p_int']).reset_index(drop = True)

    df['gn'] = -1
    group_num = 0
    assigned = np.zeros(len(df), dtype = bool)

    ir_arr = df['ir'].values
    mp_arr = df['m_p_int'].values
    val_arr = df['val'].values
    N = len(df)
    search_window = 5000

    for i in range(N):
        if assigned[i]:
            continue

        group_num += 1
        assigned[i] = True
        df.at[i, 'gn'] = group_num

        current_balance = val_arr[i]
        grp_min_mp = mp_arr[i]
        grp_max_mp = mp_arr[i]
        base_ir = ir_arr[i]

        while True:
            best_candidate_idx = -1
            best_new_balance_abs = abs(current_balance)
            best_min_mp, best_max_mp = grp_min_mp, grp_max_mp

            limit = min(N, i + search_window)

            for j in range(i + 1, limit):
                if assigned[j]:
                    continue

                if ir_arr[j] - base_ir > 0.150000001:
                    break

                new_min = min(grp_min_mp, mp_arr[j])
                new_max = max(grp_max_mp, mp_arr[j])

                if (new_max - new_min) <= 30:
                    new_balance = current_balance + val_arr[j]
                    if abs(new_balance) < best_new_balance_abs:
                        best_new_balance_abs = abs(new_balance)
                        best_candidate_idx = j
                        best_min_mp = new_min
                        best_max_mp = new_max

            if best_candidate_idx != -1:
                assigned[best_candidate_idx] = True
                df.at[best_candidate_idx, 'gn'] = group_num
                current_balance += val_arr[best_candidate_idx]
                grp_min_mp = best_min_mp
                grp_max_mp = best_max_mp

                if abs(current_balance) < 1e-5:
                    break
            else:
                break

    df = df.drop(columns = ['m_p_int', 'val'])
    df.to_csv("2b_result.csv", index = False, sep = ',')



solve_2b()