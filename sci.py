import scipy.sparse as sp
import numpy as np
import torch
from updateAdjMat import getActionNbghs
opIDsOnMchs = np.array([[7, 29, 33, 16, -6, -6],
                        [6, 18, 28, 34, 2, -6],
                        [26, 31, 14, 21, 11, 1],
                        [30, 19, 27, 13, 10, -6],
                        [25, 20, 9, 15, -6, -6],
                        [24, 12, 8, 32, 0, -6]])
action = 7 
number_of_jobs = 3
number_of_machines = 3
number_of_tasks = number_of_jobs * number_of_machines
flag = True
# LIL形式のスパース行列を作成（隣接行列として利用）
adj_matrix = sp.lil_matrix((number_of_tasks, number_of_tasks))  # 5x5の隣接行列
first_col = np.arange(start=0, stop=number_of_tasks, step=1).reshape(number_of_jobs, -1)[:, 0]
# エッジを追加

for i in range(number_of_tasks):
    adj_matrix[i, i] = 1.0
    if i < number_of_tasks-1 and i not in first_col:
        adj_matrix[i, i-1] = 1.0

# 必要に応じてCOO形式やCSR形式に変換
adj = adj_matrix.tocoo()
# エッジを削除
#adj_matrix[0, 1] = 0.0
adj = adj_matrix.tolil()
precd, succd = getActionNbghs(action, opIDsOnMchs)
adj[action,action] = 0
print("スパース隣接行列 (LIL形式):\n", adj)
exit()
adj[action, action] = 1
if action not in first_col:
    adj[action, action - 1] = 1
adj[action, precd] = 1
adj[succd, action] = 1
if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
    adj[succd, precd] = 0
           

print("スパース隣接行列 (COO形式):\n", adj)
