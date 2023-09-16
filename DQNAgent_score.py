import numpy as np
import torch
from Reversi_score import Reversi
from State_score import State
from DQN_1_layer import *

rundom_start = 0

class DQNAgent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train_mode(train)
        self.player = player
        self.env : Reversi = env

    def train_mode (self, train):
          self.train = train
          if train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state: State, epoch = 0, events= None, train = True, graphics = None):
        # self.train_mode(train)
        actions = state.legal_actions
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if np.count_nonzero(state.board)<rundom_start:
                return random.choice(actions)
            if rnd < epsilon:
                return random.choice(actions)
        
        state_tensor, action_tensor = state.toTensor()
        # action_tensor = torch.from_numpy(np.array(actions))
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1))
        
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_actions (self, states_tensor, dones):
        actions = []
        boards_tensor = states_tensor[0]
        actions_tensor = states_tensor[1]
        for i, board in enumerate(boards_tensor):
            if dones[i].item():
                actions.append((0,0))
            else:
                actions.append(self.get_Action(State.tensorToState(state_tensor=(boards_tensor[i],actions_tensor[i])), train=False))
        return torch.tensor(actions)

    def loadModel (self, file):
        self.model = torch.load(file)
    
    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None) -> Any:
        return self.get_Action(state)
