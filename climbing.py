import pandas as pd
import numpy as np
import random as rd
import sklearn
import xlrd

df = pd.read_excel('data2.xlsx')
df = sklearn.utils.shuffle(df)
df = df.reset_index(drop=True)
np.set_printoptions(suppress=True)
df = df.to_numpy()
tasks = df.shape[0]
number_of_machines = df.shape[1] - 1


def calculate(arr):
    first_row = arr[0, 0:number_of_machines]
    w = 0
    row_sum = np.array([])
    for j in first_row:
        if w == 0:
            row_sum = np.append(row_sum, j)
        elif w == 1:
            row_sum = np.append(row_sum, j)
        elif w < number_of_machines+1:
            row_sum = np.append(row_sum, j + row_sum[w - 1])
        w = w + 1
    result = row_sum

    if arr.ndim > 1:
        row_before = result
        number_of_tasks = arr.shape[0]
        df_copy = arr[1:number_of_tasks, 0:number_of_machines]
        n = 0
        for i in df_copy:
            row_add = np.array([])
            for k in i:
                if n == 0:
                    row_add = np.append(row_add, k)
                elif n == 1:
                    row_add = np.append(row_add, k + row_before[1])
                elif n < number_of_machines+1:
                    add = max(row_add[n - 1], row_before[n]) + k
                    row_add = np.append(row_add, add)
                else:
                    pass
                n = n + 1
                if n == number_of_machines:
                    n = 0
                    row_before = row_add.copy()
            result = np.vstack([result, row_add])
    return result

def random():
    a = rd.randint(0, tasks - 1)
    b = rd.randint(0, tasks - 1)
    while a == b:
        b = rd.randint(0, tasks - 1)
    return a, b

def swap(arr, a, b):
    arr[[a, b]] = arr[[b, a]]
    return arr


def main():
    dfcopy = df.copy()
    res1 = calculate(df)[-1, -1]
    rand1, rand2 = random()
    swap(df, rand1, rand2)
    res2 = calculate(df)[-1, -1]
    print(res1, res2)
    if res1 > res2:
        return df
    else:
        return dfcopy

for i in range(1000):
    final = main()
    df = final.copy()

print(calculate(final))
final = pd.DataFrame(calculate(final))
final = final.astype(int)
final.to_csv("climbing_result.csv", index=False, header=None)
