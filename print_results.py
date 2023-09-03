import numpy as np
import torch
import matplotlib.pyplot as plt


path1 = 'Data/results_fix_2.pth'
path2 = 'Data/results_fix_3.pth'

path3 = 'Data/results_fix_4.pth'
random_results_path_3 = 'Data/random_fix_4.pth'
# path2 = 'Data/results_fix_Random_start_1000k.pth'
# path3 = 'Data/results_fix_random_start_1_layer2.pth'
path4 = 'Data/results_fix_5.pth'
random_results_path_4 = 'Data/random_fix_5.pth'

path5 = 'Data/results_fix_6.pth'
random_results_path_5 = 'Data/random_fix_6.pth'

path6 = 'Data/results_fix_7.pth'
random_results_path_6 = 'Data/random_fix_7.pth'


results1 = torch.load(path1)
results2 = torch.load(path2)

results3 = torch.load(path3)
random_results_3 = torch.load(random_results_path_3)

results4 = torch.load(path4)
random_results_4 = torch.load(random_results_path_4)

results5 = torch.load(path5)
random_results_5 = torch.load(random_results_path_5)

results6 = torch.load(path6)
random_results_6 = torch.load(random_results_path_6)

# print (results1)
print(path1, max(results1['results']), np.argmax(results1['results']), len(results1['results']) )
print(path2, max(results2['results']), np.argmax(results2['results']), len(results2['results']) )
print(path3, max(results3['results']), np.argmax(results2['results']), len(results3['results']) )
print(path4, max(results4['results']), np.argmax(results4['results']), len(results4['results']) )
print(path5, max(results5['results']), np.argmax(results5['results']), len(results5['results']) )
print(path6, max(results6['results']), np.argmax(results6['results']), len(results6['results']) )


results1['avglosses'] = list(filter(lambda k:  0< k <= 0.08, results1['avglosses'] ))
results2['avglosses'] = list(filter(lambda k:  0< k <= 0.08, results2['avglosses'] ))
results3['avglosses'] = list(filter(lambda k:  0< k <= 0.08, results3['avglosses'] ))
results4['avglosses'] = list(filter(lambda k:  0< k <= 0.08, results4['avglosses'] ))
results5['avglosses'] = list(filter(lambda k:  0< k <= 0.08, results5['avglosses'] ))
results6['avglosses'] = list(filter(lambda k:  0< k <= 0.08, results6['avglosses'] ))

with torch.no_grad():
    
    fig1, ax_list1 = plt.subplots(2,2, figsize=(12,8),)
    ax_list1[0,0].plot(results1['results'])
    ax_list1[0,1].plot(results1['avglosses'])
    ax_list1[1,0].plot(results2['results'])
    ax_list1[1,1].plot(results2['avglosses'])
        
    fig2, ax2_list2 = plt.subplots(3,1)
    fig2.suptitle(path3)
    ax2_list2[0].plot(results3['results'])
    ax2_list2[1].plot(random_results_3)
    ax2_list2[2].plot(results3['avglosses'])

    fig3, ax2_list3 = plt.subplots(3,1)
    fig3.suptitle(path4)
    ax2_list3[0].plot(results4['results'])
    ax2_list3[1].plot(random_results_4)
    ax2_list3[2].plot(results4['avglosses'])
    
    fig4, ax2_list4 = plt.subplots(3,1)
    fig4.suptitle(path5)
    ax2_list4[0].plot(results5['results'])
    ax2_list4[1].plot(random_results_5)
    ax2_list4[2].plot(results5['avglosses'])
    
    fig5, ax2_list5 = plt.subplots(3,1)
    fig5.suptitle(path6)
    ax2_list5[0].plot(results6['results'])
    ax2_list5[1].plot(random_results_6)
    ax2_list5[2].plot(results6['avglosses'])
    

    plt.show()
