from random import randint
from math import comb
import pandas as pd
import sys

def is_valid(csv_file):
    EPS = 1e-9
    df = pd.read_csv(csv_file, dtype={'m_p': str})
    df.loc[df['m_p'].str.contains('<'), 'm_p'] = "0"
    df['m_p'] = df['m_p'].astype(int)
    df['ir'] = df['ir'].astype(float)
    # print(df['gn'].unique())
    for gn in df['gn'].unique():
        if gn < 0:
            continue
        part = df[df['gn'] == gn]
        days_fit = part['m_p'].max() - part['m_p'].min() <= 30
        ir_fit = part['ir'].max() - part['ir'].min() <= 0.15 + EPS
        if not (days_fit and ir_fit):
            print(gn, days_fit, ir_fit, part['m_p'].max(), part['m_p'].min(), part['ir'].max(), part['ir'].min())
            print(part)
            return gn, False
    return -1, True


def RWA(csv_file):
    EPS = 1e-9
    df = pd.read_csv(csv_file, dtype={'m_p': str})
    df.loc[df['m_p'].str.contains('<'), 'm_p'] = "0"
    df['m_p'] = df['m_p'].astype(int)
    df['ir'] = df['ir'].astype(float)

    gross, net = 0.1, 0.4
    ans = 0

    # RWA for netted groups
    for gn in df['gn'].unique():
        if gn >= 0:
            part = df[df['gn'] == gn]
            ans += abs(part['asset'].sum() - part['liab'].sum())

    # RWA for unnetted deals
    part = df[df['gn'] == -1]
    if len(part) > 0:
        tot_assets, tot_liabs = part['asset'].sum(), part['liab'].sum()
        ans += (tot_assets + tot_liabs) * gross + abs(tot_assets - tot_liabs) * net
    return ans


def single_binom(N):
    '''
    Compute the Expected Value for single_binomial distribution
    '''
    EV = 0
    for A in range(N + 1):
        EV += comb(N, A) * abs(B - A)
    return 2 * EV / (2 ** (2 * N))

def multibinom(N):
    '''
    Compute the Expected Value for multibinomial distribution
    '''
    EV = 0
    for A in range(N + 1):
        for B in range(N + 1):
            EV += comb(N, A) * comb(N, B) * abs(B - A)
    return 2 * EV / (2 ** (2 * N))


# print(sys.argv)
print(is_valid("2a_result.csv"))
print(RWA("2a_result.csv"))



print(is_valid("2b_result.csv"))
print(RWA("2b_result.csv"))