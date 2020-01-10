import pandas as pd
import xlrd
import numpy as np



df = pd.read_excel('data.xlsx')

df['sum'] = df.sum(axis=1)
df = df.sort_values(by=['sum'], ascending=False)
df = df.reset_index(drop=True)
np.set_printoptions(suppress=True)
print(df.head(5))
df = df.to_numpy()

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




