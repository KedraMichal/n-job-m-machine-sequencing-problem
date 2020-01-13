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
    df_copy = arr[1:number_of_tasks, 0:11]
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



def find(df_before, new_row):
    rows = df_before.shape[0]

    if df_before.ndim == 1:
        rows= 1
    else:
        splitted = np.vsplit(df_before, rows)

    print(rows)

    list = []
    for i in range(rows+1):
        if i == 0:
            test_arr = np.vstack([df_before,new_row])
            result_of_comb = calculate(test_arr, rows+1)[-1][-1]
            list.append(result_of_comb)
        elif i == rows:
            test_arr = 0
            test_arr = np.vstack([new_row, df_before])
            result_of_comb = calculate(test_arr, rows+1)[-1][-1]
            list.append(result_of_comb)
            print(calculate(test_arr, rows + 1))
        else:
            first_split = np.array(splitted[0:i])
            second_split = np.array(splitted[i:rows])
            dim1 = first_split[:,0,:]
            dim2 = second_split[:,0,:]
            first_stack = np.vstack([dim1, new_row])
            test_arr = np.vstack([first_stack, dim2])
            result_of_comb = calculate(test_arr, rows + 1)[-1, -1]
            list.append(result_of_comb)
            print(calculate(test_arr, rows + 1))

        if result_of_comb == min(list):
            order = test_arr
            arr_best = calculate(test_arr, rows+1)
            print(calculate(test_arr, rows + 1))
    print(list)
    return arr_best, order



for i in range(49):
    arr_start = df[0:1, 0:11]
    arr_add = df[i+1, 0:11]
    if i == 0:
        new_arr = find(arr_start, arr_add)[1]
    else:
        new_arr = find(new_arr, arr_add)[1]

    if i>0:
        print(calculate(new_arr, i+2))
        print(calculate(new_arr, i+2)[-1][-1])


