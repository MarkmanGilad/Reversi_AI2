import numpy as np
from Reversi import Reversi
from State import State
env = Reversi()
matrix = np.array([ [0, 1, 2, 0, 1, 2, 0, 1],
                    [2, 0, 1, 2, 0, 1, 2, 0],
                    [1, 2, 0, 1, 2, 0, 1, 2],
                    [0, 1, 2, 0, 1, 2, 0, 1],
                    [2, 0, 1, 2, 0, 1, 2, 0],
                    [1, 2, 0, 1, 2, 0, 1, 2],
                    [0, 1, 2, 0, 1, 2, 0, 1],
                    [2, 0, 1, 2, 0, 1, 2, 0]])

state = State(matrix)
# print(matrix)

# start_row =2
# start_col = 1

# rows1 = np.arange(start_row, 8, 1) # rows down
# cols1 = np.arange(start_col, 8, 1) # cols right
# rows2 = np.arange(start_row, -1, -1) # rows up
# cols2 = np.arange(start_col, -1, -1) # cols left
# print (rows1, cols1)
# print (rows2, cols2)

# print(matrix[rows1[:len(cols1)], cols1[:len(rows1)]]) # down-right
# print(matrix[rows2[:len(cols2)], cols2[:len(rows2)]]) # up-left
# print(matrix[rows1[:len(cols2)], cols2[:len(rows1)]]) # down-left
# print(matrix[rows2[:len(cols1)], cols1[:len(rows2)]]) # up-rught
# print(matrix[start_row, start_col:]) # right
# print(matrix[start_row, start_col::-1]) # left
# print(matrix[start_row:, start_col]) #down
# print(matrix[start_row::-1, start_col]) # up

# arr = np.array([2, 0, 1, 2, 0, 1, 2])
# player2 = np.where(arr == 2)[0]
# player1 = np.where(arr == 1)[0]
# print(player2, player1)
# print("index", player2[0],player1[1])
# arr1 = arr[player2[0]+1:player1[1]]
# print(arr1)
# print(np.all(arr1 == 1))
# rows, cols = np.where(matrix==1)
# print(rows, cols)

action = (2,4)
print(env.state.board)
print(env.is_legal_move(state = env.state, row_col= action))
print (env.move(action, env.state))
print(env.state.board)
# env.change_adjacent(env.state, action)
# print(env.state.board)
# env.update_legal(env.state)
# print(env.state.board)
s = env.get_next_state((2,5), env.state)
print(s.board)