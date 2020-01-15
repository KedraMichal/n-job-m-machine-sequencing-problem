import pandas as pd
import xlrd
import numpy as np
import random as rd
import sklearn

df = pd.read_excel('data.xlsx')
df = sklearn.utils.shuffle(df)
df = df.reset_index(drop=True)
np.set_printoptions(suppress=True)
df = df.to_numpy()
count_rows = df.shape[0]


def calculate(arr):
    first_row = arr[0, 0:11]
    w = 0
    row_sum = np.array([])
    for j in first_row:
        if w == 0:
            row_sum = np.append(row_sum, j)
        elif w == 1:
            row_sum = np.append(row_sum, j)
        elif w < 11:
            row_sum = np.append(row_sum, j + row_sum[w - 1])
        w = w + 1
    result = row_sum

    if arr.ndim > 1:
        row_before = result
        number_of_tasks = arr.shape[0]
        df_copy = arr[1:number_of_tasks, 0:11]
        n = 0
        for i in df_copy:
            row_add = np.array([])
            for k in i:
                if n == 0:
                    row_add = np.append(row_add, k)
                elif n == 1:
                    row_add = np.append(row_add, k + row_before[1])
                elif n < 11:
                    add = max(row_add[n - 1], row_before[n]) + k
                    row_add = np.append(row_add, add)
                else:
                    pass
                n = n + 1
                if n == 11:
                    n = 0
                    row_before = row_add.copy()
            result = np.vstack([result, row_add])
    return result


def swap(arr, a, b):
    arr[[a, b]] = arr[[b, a]]
    return arr


def sasiedztwo():
    list = []
    while len(list)<6: #maksymalna dlugosc sasiedztwa
        random = rd.randint(0, count_rows-1)
        if random not in list:
            list.append(random)
    return list


temp = 11
def templow():
    global temp
    temp =temp * 0.99


def main(data):
    data_copy = data.copy()
    min1 = calculate(data_copy)[-1][-1]
    s = sasiedztwo()
    min_sas = []
    for i in range(4):# dlugosc sasiedztwa
        swap(data, s[0], s[i+1])
        wynik = calculate(data)[-1][-1]
        min_sas.append(wynik)
        swap(data, s[0], s[i+1])
    print(min_sas)
    best_sas = min(min_sas) #najlepszy wynik z sasiedztwa
    best_index = min_sas.index(min(min_sas)) #indeks najlepszego rozw w sasiedztwie
    swap(data, s[0], s[best_index+1])
    data_copy2 = data.copy()

    prob = 1/(np.exp((abs(min1-best_sas))/temp))
    w = rd.random()
    print(min1, best_sas, prob, w)
    templow()
    if min1 < best_sas and prob<w:
        return data_copy
    else:
        return data_copy2

for i in range(1000):
    final = main(df)
    df = final.copy()
print(df)




