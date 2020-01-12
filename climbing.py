import pandas as pd
import xlrd
import numpy as np
import random as rd

df = pd.read_excel('data.xlsx')
df['sum'] = df.sum(axis=1)
df = df.sort_values(by=['sum'], ascending=False)
df = df.reset_index(drop=True)
np.set_printoptions(suppress=True)
print(df.head(5))
df = df.to_numpy()
count_rows = df.shape[0]


def random():
    a = rd.randint(0, count_rows - 1)
    b = rd.randint(0, count_rows - 1)
    while a == b:
        b = rd.randint(0, count_rows - 1)
    return a, b


def swap(arr, a, b):
    arr[[a, b]] = arr[[b, a]]
    return arr


def calculate(arr, number_of_tasks):
    n = 0
    first_row = arr[0, 0:11]
    x1 = arr[0, 0:11]
    df_copy = arr.copy()
    df_copy = df[1:number_of_tasks, 0:11]
    for i in df_copy:
        row_add = np.array([])
        for k in i:
            if n == 0:
                row_add = np.append(row_add, k)
            elif n == 1:
                row_add = np.append(row_add, k + first_row[1])
            elif n < 11:
                add = max(row_add[n - 1], first_row[n]) + k
                row_add = np.append(row_add, add)
            else:
                pass
            n = n + 1
            if n == 11:
                n = 0
                first_row = row_add.copy()
        x1 = np.vstack([x1, row_add])
    return x1


def main():
    dfcopy = df.copy()
    res1 = calculate(df, 50)[-1, -1]
    rand1, rand2 = random()
    swap(df, rand1, rand2)
    res2 = calculate(df, 50)[-1, -1]
    print(res1, res2)
    if res1 > res2:
        return df
    else:
        return dfcopy


for i in range(50000):
    final = main()
    df = final.copy()

print(calculate(final,50))
