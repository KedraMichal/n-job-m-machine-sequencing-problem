import pandas as pd
import numpy as np
import random as rd
import sklearn
import xlrd

df = pd.read_excel('data.xlsx')
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



list_of_pop = np.array([])
for i in range(10):
    osobnik = np.random.choice(np.arange(0,50), 50, replace=False)
    if i== 0:
        list_of_pop = np.append(list_of_pop, osobnik)
    else:
        list_of_pop = np.vstack([list_of_pop, osobnik])


print(list_of_pop)
indeks_order = list_of_pop[0,:]
indeks_order = indeks_order.astype(np.int64)
print(indeks_order)
df = df.take(indeks_order, axis=0)
print(df)
lis = np.array([])
for i in range(10):
    indeks_order = list_of_pop[0, :]
    indeks_order = indeks_order.astype(np.int64)
    df = df.take(indeks_order, axis=0)
    lis = np.append(lis, calculate(df)[-1][-1])
print(lis)
