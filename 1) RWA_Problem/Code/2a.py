import pandas as pd
import numpy as np


def solve_2a():
    df = pd.read_csv("2a.csv", sep = ';')
    df = df.iloc[:, :6]
    df.columns = ["asset", "liab", "m_d", "m_p", "ir", "gn"]

    df['asset'] = df['asset'].astype(float)
    df['liab'] = df['liab'].astype(float)
    df['ir'] = df['ir'].astype(float)
    df['val'] = df['asset'] - df['liab']

    df = df.sort_values(by = 'ir').reset_index(drop = True)

    n = len(df)
    dp = np.full(n + 1, np.inf)
    dp[n] = 0
    path = np.zeros(n, dtype = int)
    vals = df['val'].values
    irs = df['ir'].values

    limit_search = 2000

    for i in range(n - 1, -1, -1):
        current_sum = 0.0
        base_ir = irs[i]
        limit = min(n, i + limit_search)
        best_cost = np.inf
        best_j = i

        for j in range(i, limit):
            if irs[j] - base_ir > 0.150000001:
                break

            current_sum += vals[j]
            cost = abs(current_sum) + dp[j + 1]

            if cost < best_cost:
                best_cost = cost
                best_j = j

        dp[i] = best_cost
        path[i] = best_j

    df['gn'] = -1
    group_id = 0
    curr = 0
    while curr < n:
        group_id += 1
        end_idx = path[curr]
        df.loc[curr: end_idx, 'gn'] = group_id
        curr = end_idx + 1

    df = df.drop(columns = ['val'])
    df.to_csv("2a_result.csv", index = False, sep = ',')



solve_2a()