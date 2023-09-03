from RandomAgent import RandomAgent
from FixAgent import FixAgent
from FixAgent2 import FixAgent2
# from DQNAgent import DQNAgent
from DQNAgent_one_layer import DQNAgent
from Reversi import Reversi

class Tester:

    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < games_num:
            action = player.get_Action(state=env.state, train = False)
            env.move(action, env.state)
            player = self.switchPlayers(player)
            if env.is_end_of_game(env.state):
                score = env.state.score()
                if score > 0:
                    player1_win += 1
                else:
                    player2_win += 1
                env.state = env.get_init_state()
                games += 1
                player = self.player1
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    env = Reversi()
    player2 = FixAgent(env, player=2)
    # player2 = RandomAgent(env)
    path = 'Data/best_random_fix_4.pth'
    player1 = DQNAgent(player=1, parametes_path=path,train=False, env=env)
    # player1 = RandomAgent(env)
    test = Tester(env,player1, player2)
    print(test.test(10))
    
