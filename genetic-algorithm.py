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
    indeks_order = indeks_order.astype(np.int64)
    data = data.take(indeks_order, axis=0)
    res = calculate(data)[-1][-1]
    return res


def turniej(data, population):
    best_pop = np.array([])

    for i in range(len(population)):
        random_osob = np.random.choice(np.arange(0, len(population)), 2, replace=False)
        osobnik1 = population[random_osob[0], :]
        score1 = osobnik_cal(data, osobnik1)
        osobnik2 = population[random_osob[1], :]
        score2 = osobnik_cal(data, osobnik2)
        if score1 < score2:
            if len(best_pop) == 0:
                best_pop = osobnik1
            else:
                best_pop = np.vstack([best_pop, osobnik1])
        else:
            if len(best_pop) == 0:
                best_pop = osobnik2
            else:
                best_pop = np.vstack([best_pop, osobnik2])

    return best_pop


def ranking(data, population):
    arr = np.array([])
    for i in range(len(population)):
        data_start = data
        indeks_order = population[i, :]
        indeks_order = indeks_order.astype(np.int64)
        data_start = data_start.take(indeks_order, axis=0)
        arr = np.append(arr, calculate(data_start)[-1][-1])

    best_osob_index = np.argsort(arr)[:10]  # indeks naj osobnikow
    best = population[best_osob_index[:]]

    return best


def cross(best3):
    potomki = np.array([])
    for i in range(5):
        parent1 = best3[i]
        parent2 = best3[9-i]
        len1 = int(parent1.shape[0]/2)
        potomek1 = parent1[0:len1]
        for i in parent2:
            if i in potomek1:
                pass
            else:
                potomek1 = np.append(potomek1, i)
        if len(potomki) == 0:
            potomki = potomek1
        else:
            potomki = np.vstack([potomki, potomek1])

        potomek2 = parent2[0:len1]
        for i in parent1:
            if i in potomek2:
                pass
            else:
                potomek2 = np.append(potomek2, i)

        potomki = np.vstack([potomki, potomek2])

    return potomki


def mutate(osobnik):
    if (rd.random() >= 0.9):
        rand1, rand2 = np.random.choice(np.arange(0, 50), 2, replace=False)
        osobnik[[rand1, rand2]] = osobnik[[rand2, rand1]]

    return osobnik



pop = generate_pop(50)
t = turniej(df_start, pop)




