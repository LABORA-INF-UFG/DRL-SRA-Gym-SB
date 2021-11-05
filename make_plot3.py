from tqdm import tqdm
import consts
import matplotlib.pyplot as plt

from utils.compute_area import ComputeArea
from utils.save_results import SaveResults
import numpy as np

'''
Make plot for model eval training data
train_models.py for training body, using training_callback.py and model_eval.py for run the validation.
'''

tqdm_ = "10-10000"
t = 100 # avaliation rounds
#simulation_type = "stationary"
simulation_type = "n-stationary"
F = "_F_" + consts.F_D + "_ME"
#number of schedulers
n_sch = 3

#tqdm_e = tqdm(range(10,5991,10), desc='Time Steps', leave=True, unit=" time steps")
tqdm_e = tqdm(range(1000,23010,1000), desc='Time Steps', leave=True, unit=" time steps")

tss = []

rewards_drl = []
rewards_sch = [[] for i in range(n_sch)]
pkt_loss_drl = []
pkt_loss_sch = [[] for i in range(n_sch)]
pkt_d_drl = []
pkt_d_sch = [[] for i in range(n_sch)]

for i in tqdm_e:

    #n-stationary_F_2-2_ME_5_rounds_100_bloks_eps_lr_007_10_episodes
    f = 'history_final_p2/' + simulation_type + F + "_" + str(t) + '_rounds_' + str(consts.BLOCKS_EP) + '_bloks_eps_lr_' + consts.LR_D + '_' + str(i) + '_episodes.json'
    history = SaveResults.load_history(f)

    ## creating plot data
    tss.append(i) ## X axis

    ## drl reward
    rewards_drl.append(np.mean(history['rewards']['A2C']))
    ## schedulers rewards
    p_rewards_sch = [[] for i in range(n_sch)]
    for s,v in enumerate(history['rewards']['schedulers']):
        for si in range(n_sch):
            p_rewards_sch[si].append(np.array(v)[:, si])

    for s in range(n_sch):
        rewards_sch[s].append(np.mean(p_rewards_sch[s]))

# ploting

plot_colors = ['k', 'c', 'b']
plot_marks = ['o', 'p', '+']
plot_lines = ['--', '-.', '--']
sch_names = ['Round robin', 'Proportional fair', 'Max th']

plot_data = {}
# first: throughput
figure, axis = plt.subplots(1)

plot_data[ComputeArea.auc(x=tss,y=rewards_drl)] = [tss, rewards_drl, r'DRL-SRA$_{A2C}$', '-', '^', 'r']

for i,v in enumerate(rewards_sch):
    #axis.plot(tss, vrs, linestyle='dashdot', label=v.name)
    plot_data[ComputeArea.auc(x=tss, y=v)] = [tss, v, sch_names[i], plot_lines[i], plot_marks[i], plot_colors[i]]

# sortind plot data in order to get a plot with desc ordering
plot_data_sorted = sorted(plot_data.items(),key=lambda x: x[0], reverse=True)
# plotting sum-rate plot
for p in plot_data_sorted:
    axis.plot(tss, p[1][1], label=p[1][2], linestyle=p[1][3], marker=p[1][4], color=p[1][5])

axis.set_xlabel('Training episodes')
axis.set_ylabel('Mean sum-rate (Mbps)')
#axis.set_title('Sum-rate per episode')
axis.legend(loc=7)

plt.show()