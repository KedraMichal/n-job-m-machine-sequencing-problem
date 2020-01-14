import pandas as pd
import xlrd
import numpy as np

df = pd.read_excel('data.xlsx')

df['sum'] = df.sum(axis=1)
df = df.sort_values(by=['sum'], ascending=False)
df = df.reset_index(drop=True)
np.set_printoptions(suppress=True)
df = df.to_numpy()


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


def find(df_before, new_row):
    rows = df_before.shape[0]
    if df_before.ndim == 1:
        rows = 1
    else:
        splitted = np.vsplit(df_before, rows)
    list = []
    for i in range(rows + 1):
        if i == 0:
            test_arr = np.vstack([df_before, new_row])
            result_of_comb = calculate(test_arr)[-1][-1]
            list.append(result_of_comb)
        elif i == rows:
            test_arr = np.vstack([new_row, df_before])
            result_of_comb = calculate(test_arr)[-1][-1]
            list.append(result_of_comb)
        else:
            first_split = np.array(splitted[0:i])
            second_split = np.array(splitted[i:rows])
            dim1 = first_split[:, 0, :]
            dim2 = second_split[:, 0, :]
            first_stack = np.vstack([dim1, new_row])
            test_arr = np.vstack([first_stack, dim2])
            result_of_comb = calculate(test_arr)[-1, -1]
            list.append(result_of_comb)
        if result_of_comb == min(list):
            order = test_arr
            arr_best = calculate(test_arr)

    return arr_best, order


for i in range(49):
    arr_start = df[0:1, 0:11]
    arr_add = df[i + 1, 0:11]
    if i == 0:
        new_arr = find(arr_start, arr_add)[1]
    else:
        new_arr = find(new_arr, arr_add)[1]

    print(calculate(new_arr))
    print(calculate(new_arr)[-1][-1])

pd.DataFrame(calculate(new_arr)).to_csv("neh_result.csv", index=False, header=None)