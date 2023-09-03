import numpy as np
import torch

class State:
    def __init__(self, board= None, player = 1, legal_actions = [(-1,-1)]) -> None:
        self.board = board
        self.player = player
        self.action : tuple[int, int] = None
        self.legal_actions = legal_actions

    def get_opponent (self):
        if self.player == 1:
            return 2
        else:
            return 1

    def switch_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def score (self, player = 1):
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        player_score = np.count_nonzero(self.board == player)
        opponent_score = np.count_nonzero(self.board == opponent)
        return player_score - opponent_score

    def reward (self, player = 1):
        scores = self.score(player=player)
        return (scores[0]-scores[1])/ 10.0
        # if scores > 0:
        #     return 1
        # elif scores < 0:
        #     return -1
        # else:
        #     return 0

    def __eq__(self, other) ->bool:
        return np.equal(self.board, other.board).all() 

    def __hash__(self) -> int:
        return hash(repr(self.board))
    
    def copy (self):
        newBoard = np.copy(self.board)
        legal_actions = self.legal_actions.copy()
        return State(board=newBoard, player=self.player, legal_actions=legal_actions)
    
    def toTensor (self, device = torch.device('cpu')):
        board_np = self.board.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
        actions_np = np.array(self.legal_actions)
        actions_tensor = torch.from_numpy(actions_np)
        return board_tensor, actions_tensor
    
    [staticmethod]
    def tensorToState (state_tensor, player = 1):
        board = state_tensor[0]
        board = board.reshape([8,8]).cpu().numpy()
        legal_actions = state_tensor[1]
        legal_actions = legal_actions.cpu().numpy()
        legal_actions = list(map(tuple, legal_actions))
        return State(board, player, legal_actions=legal_actions)
