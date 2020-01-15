import pandas as pd
import numpy as np
import random as rd
import sklearn
import xlrd

df = pd.read_excel('data.xlsx')
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
        elif w < number_of_machines + 1:
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
                elif n < number_of_machines + 1:
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


def swap(arr, a, b):
    arr[[a, b]] = arr[[b, a]]
    return arr


def tabu(tabu_lis, tabu1, tabu2):
    tabu_lis.append((tabu1, tabu2))
    if len(tabu_lis) > 3:
        del tabu_lis[0]


tabu_list = []


def main(data):
    comb = []
    print("dzial")
    add_first = True
    min = 0
    print(tabu_list)
    for i in range(tasks - 1):
        for j in range(i + 1, tasks):
            x1 = data[i, 0] - 1
            x2 = data[j, 0] - 1
            swap(data, x1, x2)
            new_score = calculate(data)[-1][-1]
            comb.append(new_score)
            if (x1, x2) not in tabu_list and (x2, x1) not in tabu_list:
                if add_first is True or new_score < min:
                    add_first = False
                    first = x1
                    second = x2
                    min = new_score
            swap(data, x1, x2)

    swap(data, first, second)
    tabu(tabu_list, first, second)
    print(calculate(data)[-1][-1])
    return data


for i in range(50):
    final = main(df)
    df = final.copy()
