import numpy as np
import torch
import matplotlib.pyplot as plt

# path1 = 'Python/Reversi_AI2/Data/results_fix_3_1000k.pth'
path2 = 'Python/Reversi_AI2/Data/results_fix_Random_start_1000k.pth'
path3 = 'Python/Reversi_AI2/Data/results_fix_random_start_1_layer2.pth'
# results1 = torch.load(path1)
results2 = torch.load(path2)
results3 = torch.load(path3)
# print (results1)
# print(max(results1['results']), np.argmax(results1['results']), len(results1['results']) )
print(max(results2['results']), np.argmax(results2['results']), len(results2['results']) )
print(max(results3['results']), np.argmax(results3['results']), len(results3['results']) )

results2['avglosses'] = list(filter(lambda k:  0< k <= 0.06, results2['avglosses'] ))
results3['avglosses'] = list(filter(lambda k:  0< k <= 0.06, results3['avglosses'] ))

with torch.no_grad():
    # plt.subplot(3,2,1)
    # plt.plot(results1['results'])
    # plt.title('fixPlayer results every 100 games')
    # plt.subplot(3,2,2)
    # plt.plot(results1['avglosses'])
    # plt.title('fixPlayer average loss every 100 games')
    
    plt.subplot(2,2,1)
    plt.plot(results2['results'])
    plt.title('fix random start results every 100 games')
    plt.subplot(2,2,2)
    plt.plot(results2['avglosses'])
    plt.title('fix random start average loss every 100 games')
    
    plt.subplot(2,2,3)
    plt.plot(results3['results'])
    plt.title('One layer results every 100 games')
    plt.subplot(2,2,4)
    plt.plot(results3['avglosses'])
    plt.title('One layer average loss every 100 games')
    
    
    plt.show()
