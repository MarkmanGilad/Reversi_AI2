import numpy as np
import torch
from State_score import State
from Graphics import *


class Reversi:
    def __init__(self, state:State = None) -> None:
        if state == None:
            self.state = self.get_init_state((ROWS, COLS))
        else:
            self.state = state

    def get_init_state(self, Rows_Cols = (ROWS, COLS)):
        rows, cols = Rows_Cols
        board = np.zeros([rows, cols],int)
        board[3][3] = 1
        board[3][4] = 2
        board[4][3] = 2
        board[4][4] = 1
        legal_actions = [(3,5), (5,3), (4,2), (2, 4)]
        return State (board, player=1, legal_actions=legal_actions)

    def is_free(self, row_col: tuple[int, int], state: State):
        row, col = row_col
        return state.board[row, col] != 0

    def is_inside(self, row_col, state: State):
        row, col = row_col
        board_row, board_col = state.board.shape
        return 0 <= row < board_row and 0 <= col < board_col

    def flip_piece(self, row_col, state: State):
        row, col = row_col
        if state.board[row][col] == 1:
            state.board[row][col] = 2
        else:
            state.board[row][col] = 1

    def check_legal_line(self, start_row_col: tuple[int, int], dir_row_col: tuple[int, int], state: State):
        count = 0
        opponent = state.get_opponent()
        row, col = start_row_col
        dir_row, dir_col = dir_row_col
        go = True
        while (go):
            row += dir_row
            col += dir_col
            if self.is_inside((row, col), state) and state.board[row, col] == opponent:
                count +=1
            else:
                go = False

        if self.is_inside((row, col), state) and state.board[row, col] == state.player and count > 0:
            return count
        
        return -1

    def move(self, action: tuple[int, int], state: State):
        row, col = action
        directions = (-1 , 0 , 1)
        if state.board[row][col] !=0:
            return False
        legal = False
        for dir_row in directions:
            for dir_col in directions:
                if dir_row == dir_col == 0:
                    continue
                count = self.check_legal_line((row, col), (dir_row, dir_col), state)
                if  count > 0:
                    legal = True
                    self.reverse_line((row, col), (dir_row, dir_col), count, state)
        if legal:
            state.board[row, col] = state.player
            state.switch_player()
            state.action = action
            self.set_legal_actions(state)
        return legal

    def is_legal_move(self, row_col, state: State):
        row, col = row_col
        if state.board[row][col] !=0:
            return False
        directions = (-1 , 0 , 1)
        for dir_row in directions:
            for dir_col in directions:
                if dir_row == dir_col == 0:
                    continue
                count = self.check_legal_line((row, col), (dir_row, dir_col), state)
                if  count > 0:
                    return True
        return False

    def reverse_line (self, row_col, dir_row_col, count, state: State):
        row, col = row_col
        dir_row, dir_col = dir_row_col
        opponent = state.get_opponent()
        for i in range(count):
            row += dir_row
            col += dir_col
            self.flip_piece((row, col), state)
   
    def set_legal_actions(self, state: State):
        legal_actions = []
        rows, cols = state.board.shape
        for row in range(rows):
            for col in range(cols):
                if self.is_legal_move((row,col), state):
                    legal_actions.append((row, col))
        if len(legal_actions)==0:
            state.legal_actions = [(-1,-1)]
        else:
            state.legal_actions = legal_actions

    def is_end_of_game(self, state: State):
        # legal_moves = self.get_legal_actions(state)
        if state.legal_actions[0] != (-1,-1):
            return False
        return True

    def get_next_state(self, action, state:State):
        next_state = state.copy()
        self.move(action, next_state)
        self.set_legal_actions(next_state)
        return next_state
    
    def get_all_next_states (self, state: State):
        legal_actions = state.legal_actions
        next_states = []
        for action in legal_actions:
            next_states.append(self.get_next_state(action, state))
        return next_states, legal_actions
    
    def toTensor (self, list_states, device = torch.device('cpu')):
        list_tensors = []
        list_legal_actions = []
        for state in list_states:
            tensor, legal_actions = state.toTensor(device) 
            list_tensors.append(tensor)
            list_legal_actions.append(torch.tensor(legal_actions))
        return torch.vstack(list_tensors), torch.vstack(list_legal_actions)
    
    def reward (self, state, action = None, player = None):
        if action:
            next_state = self.get_next_state(action, state)
        else:
            next_state = state
        if player:
            player = player
        else:
            player = state.player
        if (self.is_end_of_game(next_state)):
            return next_state.reward(player), True  
        return 0, False
    
    # def afterState_reward (self, afterState):
    #     if (self.is_end_of_game(afterState)):
    #         score = afterState.score() 
    #         return score[0]-score[1], True
    #     return 0, False
