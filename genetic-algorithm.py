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


k = ranking(df_start, generate_pop(5000))
potomki_list = cross(k)
j = 0
score_arr = np.array([])
for i in potomki_list:
    potomki_list[j] = mutate(i)
    score = osobnik_cal(df_start, potomki_list[j])
    score_arr = np.append(score_arr, score)
    if score == min(score_arr):
        score_best = score
        comb_best = potomki_list[j]
    j = j+1

print(score_best)
print(comb_best)








