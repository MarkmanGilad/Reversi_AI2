from Reversi import Reversi
from MinMaxAgent import MinMaxAgent
from MinMaxAgent2 import MinMaxAgent2
from AlphBetaAgent import AlphaBetaAgent
from DQNAgent_one_layer import DQNAgent
from State import State
from RandomAgent import RandomAgent
from FixAgent import FixAgent
from FixAgent2 import FixAgent2
import torch
import numpy as np

environment = Reversi()
# player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)
# player1 = MinMaxAgent2(player = 1,depth = 3, environment=environment)
# player1 = AlphaBetaAgent(player = 1,depth = 3, environment=environment)
player1 = RandomAgent(environment)
# player1 = FixAgent(environment, player=1, train=True)
# player1 = FixAgent2(environment, player=1, train=False)

# path='Data/best_fix_3.pth'
# path='Python/Reversi - AI - Q/Data/Leaky_fix2_1000k.pth'
path = None
# player1 = DQNAgent(player=1, parametes_path=path,train=False, env=environment)

# player2 = MinMaxAgent(player = 2,depth = 3, environment=environment)
# player2 = MinMaxAgent2(player = 2,depth = 3, environment=environment)
# player2 = AlphaBetaAgent(player = 2,depth = 2, environment=environment)
player2 = RandomAgent(environment)
# player2 = FixAgent(environment, player=2, train=False)
# player2 = FixAgent2(environment, player=2,train=False)


def main ():
    player = player1
    player1_win = 0
    player2_win = 0
    games = 0
    totalScore = 0
    while games < 1000:
        action = player.get_Action(state=environment.state)
        environment.move(action, environment.state)
        player = switchPlayers(player)
        if environment.is_end_of_game(environment.state):
            score = environment.state.score()
            if score > 0:
                player1_win += 1
            else:
                player2_win += 1
            totalScore += score
            environment.state = environment.get_init_state()
            player = player1
            games += 1
            print (f"Game no.: {games}, score: {player1_win, player2_win} total score: {totalScore}",end="\r")
    if path:
        print('\n' + path)
    else:
        print()
        

def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

if __name__ == '__main__':
    main()
    
