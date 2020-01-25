import pandas as pd
import numpy as np
import random as rd
import sklearn
import xlrd

df = pd.read_excel('data.xlsx')
np.set_printoptions(suppress=True)

df = df.to_numpy()
tasks = df.shape[0]
number_of_machines = df.shape[1]
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


def generate_pop(population_size):
    list_of_pop = np.array([])
    for i in range(population_size):
        osobnik = np.random.choice(np.arange(0, tasks), tasks, replace=False)
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


def tournament(data, population):
    best_pop = np.array([])

    for i in range(len(population)):
        random_osob = np.random.choice(np.arange(0, len(population)), 2, replace=False)
        osobnik1 = population[random_osob[0]]
        score1 = osobnik_cal(data, osobnik1)
        osobnik2 = population[random_osob[1]]
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


def rank_roulette(data, population):
    scores = np.array([])
    for i in range(len(population)):
        data_start = data
        offspring_score = osobnik_cal(data_start, population[i])
        scores = np.append(scores, offspring_score)

    scores_sort = np.argsort(scores)
    population = population[scores_sort]

    marks = np.arange(1, len(population) + 1)
    sum_marks = np.sum(marks)
    prob = np.sort(marks / sum_marks)[::-1]
    index = np.arange(len(population))

    roulette = np.random.choice(index, size=len(population), replace=True, p=prob)
    selected_population = population[roulette]

    return selected_population


def crossover(parents):
    offsprings = np.array([])
    for i in range(int(len(parents) / 2)):
        parent1 = parents[i]
        parent2 = parents[len(parents) - 1 - i]
        len1 = int(parent1.shape[0] / 2)
        offspring1 = parent1[0:len1]
        for i in parent2:
            if i in offspring1:
                pass
            else:
                offspring1 = np.append(offspring1, i)
        if len(offsprings) == 0:
            offsprings = offspring1
        else:
            offsprings = np.vstack([offsprings, offspring1])

        offspring2 = parent2[0:len1]
        for i in parent1:
            if i in offspring2:
                pass
            else:
                offspring2 = np.append(offspring2, i)

        offsprings = np.vstack([offsprings, offspring2])

    return offsprings


def order_crossover(parents):
    number1 = np.random.randint(len(parents))
    number2 = np.random.randint(len(parents))
    offsprings = np.array([])
    for i in range(int(len(parents)/2)):
        if number1 <= number2:
            offspring1 = parents[i, number1:number2]
            offspring2 = parents[len(parents) - i - 1, number1:number2]
            start_len = len(parents) - number2
            start_len2 = len(parents) - number2
            add_order_offspring1 = parents[len(parents) - i - 1, number2:]
            add_order_offspring1 = np.append(add_order_offspring1, parents[len(parents) - i - 1, 0:number2])
            add_order_offspring2 = parents[i, number2:]
            add_order_offspring2 = np.append(add_order_offspring2, parents[i, 0:number2])
        else:
            offspring1 = parents[i, number2:number1]
            offspring2 = parents[len(parents) - i - 1, number2:number1]
            start_len = len(parents) - number1
            start_len2 = len(parents) - number1
            add_order_offspring1 = parents[len(parents) - i - 1, number1:]
            add_order_offspring1 = np.append(add_order_offspring1, parents[len(parents) - i - 1, 0:number1])
            add_order_offspring2 = parents[i, number1:]
            add_order_offspring2 = np.append(add_order_offspring2, parents[i, 0:number1])

        for j in add_order_offspring1:
            if j not in offspring1:
                if start_len > 0:
                    offspring1 = np.append(offspring1, j)
                    start_len = start_len - 1
                else:
                    offspring1 = np.insert(offspring1, 0, j)
        if len(offsprings) == 0:
            offsprings = offspring1
        else:
            offsprings = np.vstack([offsprings, offspring1])

        for j in add_order_offspring2:
            if j not in offspring2:
                if start_len2 > 0:
                    offspring2 = np.append(offspring2, j)
                    start_len2 = start_len2 - 1
                else:
                    offspring2 = np.insert(offspring2, 0, j)
        offsprings = np.vstack([offsprings, offspring2])

    return offsprings


def mutate(offspring_array):
    for i in range(len(offspring_array)):
        if rd.random() >= 0.95:
            rand1, rand2 = np.random.choice(np.arange(0, tasks), 2, replace=False)
            offspring_array[i, [rand1, rand2]] = offspring_array[i, [rand2, rand1]]

    return offspring_array


def save_best(population_array, data):
    data_copy = data.copy()
    first = True
    for i in population_array:
        data = data_copy
        score = osobnik_cal(data, i)
        if first:
            best_score = score
            best_order = i
            first = False
        elif (score < best_score):
            best_score = score
            best_order = i

    return best_order, score


pop = generate_pop(100)
for i in range(100):
    t = rank_roulette(df_start, pop)  # selection: "tournament" or "rank_roulette"
    w = crossover(t)  # crossover: "order_crossover" or "crossover"
    m = mutate(w)
    s = save_best(m, df_start)
    if i == 0:
        best_s = s
    elif i > 0:
        if s[1] < best_s[1]:
            best_s = s

    pop = m
    print(best_s[1])

