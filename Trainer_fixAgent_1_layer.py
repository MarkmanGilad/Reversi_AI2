from Reversi import Reversi
from State import State
from DQNAgent_one_layer import DQNAgent
from ReplayBuffer import ReplayBuffer
# from RandomAgent import RandomAgent
from FixAgent_random_start import FixAgent
import torch
# from TesterClass import Tester

epochs = 2000000
start_epoch = 0
C = 1000
learning_rate = 0.01
batch_size = 64
env = Reversi()

path_load= None
path_Save='Reversi_AI2/Data/fix_random_start_1_layer2.pth'
path_best = 'Reversi_AI2/Data/best_fix_random_start_1_layer2.pth'
buffer_path = 'Reversi_AI2/Data/buffer_fix_random_start_1_layer2.pth'
results_path='Reversi_AI2/Data/results_fix_random_start_1_layer2.pth'

def main ():
    # data = torch.load(results_path)
    player1 = DQNAgent(player=1, env=env,parametes_path=path_load)
    # player2 = RandomAgent(player=2, env=env)
    player2 = FixAgent(player=2, env=env, train=False)
    buffer = ReplayBuffer(path=None)
    Q = player1.DQN
    Q_hat = Q.copy()
    Q_hat.train = False
    results = []
    avgLosses = []
    avgLoss = 0
    loss = torch.Tensor([0])
    res = 0
    best_res = -200
    loss_count = 0
    # tester = Tester(player1=player1, player2=player2, env=env)
    # results = torch.load(results_path)
    # results = data['results']
    # avgLosses = data['avglosses']
    # avgLoss = avgLosses[-1]
    
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,1000, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*10000, 30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state = env.get_init_state()
        while not env.is_end_of_game(state):
            # Sample Environement
            action = player1.get_Action(state, epoch=epoch)
            after_state = env.get_next_state(state=state, action=action)
            reward, end_of_game = env.reward(after_state, player=player1.player)
            if end_of_game:
                res += reward
                buffer.push(state, action, reward, after_state, True)
                break
            after_action = player2.get_Action(state=after_state)
            next_state = env.get_next_state(state=after_state, action=after_action)
            reward, end_of_game = env.reward(state=next_state, player=player1.player)
            if abs(reward) == 1:
                res += reward
            buffer.push(state, action, reward, next_state, end_of_game)
            state = next_state

            if len(buffer) < 4000:
                continue
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states[0], actions)
            next_actions = player1.get_actions(next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states[0], next_actions)

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())
            scheduler.step()
            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            

        if (epoch+1) % 100 == 0:
                print(f'\nres= {res}')
                avgLosses.append(avgLoss)
                results.append(res)
                if best_res < res:
                    best_res = res
                    player1.save_param(path_best)
                    if best_res > 90:
                        break
                res = 0

        if (epoch+1) % 5000 == 0:
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses}, results_path)
            torch.save(buffer, buffer_path)
            player1.save_param(path_Save)
        if len(buffer) > 5000:
            print (f'epoch={epoch} loss={loss:.5f} Q_values[0]={Q_values[0].item():.3f} avgloss={avgLoss:.5f}', end=" ")
            print (f'learning rate={scheduler.get_last_lr()[0]} path={path_Save} res= {res} best_res = {best_res}')

    torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses}, results_path)
    torch.save(buffer, buffer_path)

if __name__ == '__main__':
    main()
