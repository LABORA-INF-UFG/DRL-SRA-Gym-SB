import os
import sys

sys.path.insert(1, "/content/drive/MyDrive/DRL-SRA-Gym/")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import ACKTR, A2C, TRPO, DQN, PPO1, PPO2, ACER
from tqdm import tqdm
import json
from json import JSONEncoder
from stable_baselines.common.policies import MlpPolicy

import consts
import copy

## env dealling with combinatorial action space
## computing percentual of the packet loss per allocation
from sra_env.sra_env7 import SRAEnv

x = list(range(1,21,1))
tqdm_e = tqdm(x, desc='Time Steps', leave=True, unit=" time steps")


t_pkt_loss_drl_all, t_pkt_loss_drl2_all = [], []
t_rw_drl_all, t_rw_drl2_all = [], []
t_pkt_d_all, t_pkt_d2_all = [], []

t_pkt_loss_sch_all = [[] for i in range(3)]
t_rw_sch_all = [[] for i in range(3)]
t_pkt_d_sch_all = [[] for i in range(3)]

## esse abaixo funciona!! ganhamos na latência com carga baixa (tit)
#teoricamente, esse prioriza a latência
model1 = PPO1.load('trained_models_4/v3/ppo1_100000_F_2-2_low_all_v2_gamma_07_lr_007_epsilon_1e-05')
# teoricamente prioriza a vazão
model2 = PPO1.load('trained_models_4/v3/ppo1_100000_F_2-2_low_tradicional_ti9_gamma_07_lr_007_epsilon_1e-05')

for ti in tqdm_e:

    env1 = SRAEnv(type="Master", running_tp=1, desc="A2C", ti=ti)

    env1.reset()
    obs, rw, endep, info = env1.step_(0)

    #building the seccond env to deal with the traditional model
    obs2 = copy.deepcopy(obs)
    env2 = copy.deepcopy(env1)
    env2.type = "Slave"
    env2.par_envs['Master'].append(env1)

    pkt_loss_drl_all, pkt_loss_drl2_all = [], []
    rw_drl_all, rw_drl2_all = [], []
    pkt_loss_sch_all = [[] for i in range(len(env1.schedulers))]
    rw_sch_all = [[] for i in range(len(env1.schedulers))]
    pkt_d_drl_all, pkt_d_drl2_all = [], []
    pkt_d_sch_all = [[] for i in range(len(env1.schedulers))]

    # 10 episodes
    for i in range(50):

        rw_sch = [[] for i in range(len(env1.schedulers))]
        rw_drl, rw_drl2 = [], []
        pkt_loss_sch = [[] for i in range(len(env1.schedulers))]
        pkt_loss_drl, pkt_loss_drl2 = [], []
        pkt_d_sch = [[] for i in range(len(env1.schedulers))]
        pkt_d_drl, pkt_d_drl2 = [], []

        # running a entire episode
        for ii in range(env1.blocks_ep):
            #first env
            action1, _ = model1.predict(obs, deterministic=True)
            obs, rewards_1, _, _ = env1.step_(action1)
            rw_drl.append(rewards_1[0])
            if rewards_1[2][0] > -10.:
                pkt_loss_drl.append(rewards_1[2][0])
            pkt_d_drl.append(np.mean(rewards_1[3][0]))
            ## running for each scheduler
            for u, v in enumerate(env1.schedulers):
                if rewards_1[2][u + 1] > -10.:
                    pkt_loss_sch[u].append(rewards_1[2][u + 1])
                rw_sch[u].append(rewards_1[1][u])
                pkt_d_sch[u].append(np.mean(rewards_1[3][u + 1]))
                # print(rw_sh)
                # print(rewards_1[2][u+1])

            #seccond env
            action2, _ = model2.predict(obs2, deterministic=True)
            obs2, rewards_2, _, _ = env2.step_(action2)
            rw_drl2.append(rewards_2[0])
            if rewards_2[2][0] > -10.:
                pkt_loss_drl2.append(rewards_2[2][0])
            pkt_d_drl2.append(np.mean(rewards_2[3][0]))


        #print(str(np.mean(pkt_loss) * 100) + ' %')
        #print("Throughput " + str(np.mean(rw_sh) / env1.K) + " Mbps")
        ## running for each scheduler
        for u, v in enumerate(env1.schedulers):
            pkt_loss_sch_all[u].append(np.mean(pkt_loss_sch[u]))
            rw_sch_all[u].append(np.mean(rw_sch[u]) / env1.K)
            pkt_d_sch_all[u].append(pkt_d_sch[u])

        # drl agents
        pkt_loss_drl_all.append(np.mean(pkt_loss_drl))
        rw_drl_all.append(np.mean(rw_drl) / env1.K)
        pkt_d_drl_all.append(pkt_d_drl)

        pkt_loss_drl2_all.append(np.mean(pkt_loss_drl2))
        rw_drl2_all.append(np.mean(rw_drl2) / env2.K)
        pkt_d_drl2_all.append(pkt_d_drl2)



    # compute the parameter averages for the completed episode
    ## running for each scheduler
    for u, v in enumerate(env1.schedulers):
        t_pkt_loss_sch_all[u].append(np.mean(pkt_loss_sch_all[u]))
        t_rw_sch_all[u].append(np.mean(rw_sch_all[u]))
        t_pkt_d_sch_all[u].append(np.mean(pkt_d_sch_all[u]))
    # drl agents
    t_pkt_loss_drl_all.append(np.mean(pkt_loss_drl_all))
    t_rw_drl_all.append(np.mean(rw_drl_all))
    t_pkt_d_all.append(np.mean(pkt_d_drl_all))

    t_pkt_loss_drl2_all.append(np.mean(pkt_loss_drl2_all))
    t_rw_drl2_all.append(np.mean(rw_drl2_all))
    t_pkt_d2_all.append(np.mean(pkt_d_drl2_all))

figure, axis1 = plt.subplots(1)

for u, v in enumerate(env1.schedulers):
    axis1.plot(x, t_pkt_loss_sch_all[u], label=v.name)
axis1.plot(x, t_pkt_loss_drl_all, label='DRL-SRA$_{PPO1_{Delay}}$')
axis1.plot(x, t_pkt_loss_drl2_all, label='DRL-SRA$_{PPO1_{SR}}$')
axis1.set_xlabel('Mean user traffic load (Mbps)')
axis1.set_ylabel('Mean user packet loss (%)')
#axis2.set_title('Packet loss per episode')
axis1.legend(loc=4)

figure2, axis2 = plt.subplots(1)

for u, v in enumerate(env1.schedulers):
    axis2.plot(x, t_pkt_d_sch_all[u], label=v.name)
axis2.plot(x, t_pkt_d_all, label='DRL-SRA$_{PPO1_{Delay}}$')
axis2.plot(x, t_pkt_d2_all, label='DRL-SRA$_{PPO1_{SR}}$')
axis2.set_xlabel('Mean user traffic load (Mbps)')
axis2.set_ylabel('BS buffering average latency')
#axis2.set_title('Packet loss per episode')
axis2.legend(loc=4)

figure3, axis3 = plt.subplots(1)

for u, v in enumerate(env1.schedulers):
    axis3.plot(x, t_rw_sch_all[u], label=v.name)
axis3.plot(x, t_rw_drl_all, label='DRL-SRA$_{PPO1_{Delay}}$')
axis3.plot(x, t_rw_drl2_all, label='DRL-SRA$_{PPO1_{SR}}$')
axis3.set_xlabel('Mean user traffic load (Mbps)')
axis3.set_ylabel('Mean user throughput (Mbps)')
#axis2.set_title('Packet loss per episode')
axis3.legend(loc=4)

plt.show()
