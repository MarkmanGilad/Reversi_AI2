from Reversi import Reversi
from State import State
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from RandomAgent import RandomAgent
import torch
from TesterClass import Tester

epochs = 1000000
C = 1000
learning_rate = 0.1
batch_size = 32
env = Reversi()

path_load= None
path_Save='Data/Leaky_1000k.pth'
buffer_path = 'Data/buffer_Leaky_1000k.pth'
results_path='Data/results_Leaky_1000k.pth'

def main ():
    
    player1 = DQNAgent(player=1, env=env,parametes_path=path_load)
    player2 = RandomAgent(player=2, env=env)
    buffer = ReplayBuffer(path=None)
    Q = player1.DQN
    Q_hat = Q.copy()
    Q_hat.train = False
    tester = Tester(player1=player1, player2=player2, env=env)
    # results = torch.load(results_path)
    results = []
    # print (results)
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,10000, gamma=0.9)
    for epoch in range(0, epochs):
        
        state = env.get_init_state()
        while not env.is_end_of_game(state):
            # Sample Environement
            action = player1.get_Action(state, epoch=epoch)
            after_state = env.get_next_state(state=state, action=action)
            reward, end_of_game = env.reward(after_state, player=player1.player)
            if end_of_game:
                buffer.push(state, action, reward, after_state, True)
                break
            after_action = player2.get_Action(state=after_state)
            next_state = env.get_next_state(state=after_state, action=after_action)
            reward, end_of_game = env.reward(state=next_state, player=player1.player)
            buffer.push(state, action, reward, next_state, end_of_game)
            state = next_state
                        
            if len(buffer) < 640:
                continue
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states, actions)
            next_actions = player1.get_actions(next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions)

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())
            # scheduler.step()

        if epoch % 5000 == 0:
            res = tester(100)
            results.append(res)            
            print(res)
            player1.save_param(path_Save)
            torch.save(results, results_path)
            torch.save(buffer, buffer_path)
        if epoch > 22:
            print (epoch, loss, Q_values[0], end="\r")

    player1.save_param(path_Save)
    torch.save(results, results_path)
    torch.save(buffer, buffer_path)

if __name__ == '__main__':
    main()
