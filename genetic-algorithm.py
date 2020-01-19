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
df_start = df.copy()
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


def generate_pop(population_size):
    list_of_pop = np.array([])
    for i in range(population_size):
        osobnik = np.random.choice(np.arange(0, 50), 50, replace=False)
        if i == 0:
            list_of_pop = np.append(list_of_pop, osobnik)
        else:
            list_of_pop = np.vstack([list_of_pop, osobnik])
    return list_of_pop


def osobnik_cal(data, indeks_order):
    d_copy = data
    indeks_order = indeks_order.astype(np.int64)
    d_copy = d_copy.take(indeks_order, axis=0)
    res = calculate(d_copy)[-1][-1]
    return res


def ranking(data, population):
    arr = np.array([])
    for i in range(len(population)):
        data_start = data
        indeks_order = population[i, :]
        indeks_order = indeks_order.astype(np.int64)
        data_start = data_start.take(indeks_order, axis=0)
        arr = np.append(arr, calculate(data_start)[-1][-1])

    best_osob_index = np.argsort(arr)[:2]  # indeks naj osobnikow
    best1 = population[best_osob_index[0]]
    best2 = population[best_osob_index[1]]

    return best1, best2


def cross(parent1, parent2):
    len1 = int(parent1.shape[0] / 2)
    potomek1 = parent1[0:len1]
    for i in parent2:
        if i in potomek1:
            pass
        else:
            potomek1 = np.append(potomek1, i)
    potomek2 = parent2[0:len1]
    for i in parent1:
        if i in potomek2:
            pass
        else:
            potomek2 = np.append(potomek2, i)

    return potomek1, potomek2


def mutate(osobnik):
    if (rd.random() >= 0.9):
        rand1, rand2 = np.random.choice(np.arange(0, 50), 2, replace=False)
        osobnik[[rand1, rand2]] = osobnik[[rand2, rand1]]

    return osobnik



k = ranking(df_start, generate_pop(1000))
print(osobnik_cal( df_start, k[0]), osobnik_cal(df_start, k[1]))
f, g = cross(k[0], k[1])
print(osobnik_cal( df_start, f), osobnik_cal(df_start, g))
mutate(f)
mutate(g)
print(osobnik_cal( df_start, f), osobnik_cal(df_start, g))
