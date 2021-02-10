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


t_pkt_loss_drl_all = []
t_rw_drl_all = []
t_pkt_loss_all = []
t_rw_sh_all = []
t_pkt_d_all = []
t_pkt_d_sch_all = []

## esse abaixo funciona!! ganhamos na latência com carga baixa (tit)
model1 = PPO1.load('trained_models_4/v3/ppo1_100000_F_2-2_low_all_v2_gamma_07_lr_007_epsilon_1e-05')


for ti in tqdm_e:

    env1 = SRAEnv(type="Master", running_tp=1, desc="A2C", ti=ti)

    env1.reset()
    obs, rw, endep, info = env1.step_(0)

    pkt_loss_drl_all = []
    rw_drl_all = []
    pkt_loss_all = []
    rw_sh_all = []
    pkt_d_all = []
    pkt_d_sch_all = []

    # 10 episodes
    for i in range(10):
        pkt_loss = []
        rw_sh = []
        rw_drl = []
        pkt_loss_drl = []
        pkt_d_drl = []
        pkt_d_sch = []
        # running a entire episode
        for ii in range(env1.blocks_ep):
            action1, _ = model1.predict(obs, deterministic=True)
            obs, rewards_1, _, _ = env1.step_(action1)
            rw_drl.append(rewards_1[0])
            if rewards_1[2][0] > -10.:
                pkt_loss_drl.append(rewards_1[2][0])
            pkt_d_drl.append(np.mean(rewards_1[3][0]))
            ## running for each scheduler
            for u, v in enumerate(env1.schedulers):
                if rewards_1[2][u + 1] > -10.:
                    pkt_loss.append(rewards_1[2][u + 1])
                rw_sh.append(rewards_1[1][u])
                pkt_d_sch.append(np.mean(rewards_1[3][u + 1]))
                # print(rw_sh)
                # print(rewards_1[2][u+1])

        #print(str(np.mean(pkt_loss) * 100) + ' %')
        #print("Throughput " + str(np.mean(rw_sh) / env1.K) + " Mbps")
        pkt_loss_all.append(np.mean(pkt_loss))
        rw_sh_all.append(np.mean(rw_sh) / env1.K)
        pkt_loss_drl_all.append(np.mean(pkt_loss_drl))
        rw_drl_all.append(np.mean(rw_drl) / env1.K)

        pkt_d_all.append(pkt_d_drl)
        pkt_d_sch_all.append(pkt_d_sch)

    #print("---------------TI= " + str(ti))

    #print("-----------------------------")
    # print(str(np.mean(pkt_loss_all) * 100) + ' %')
    # print("Baseline Throughput " + str(np.mean(rw_sh_all)) + " Mbps")
    # print("Delay " + str(np.mean(pkt_d_sch_all)))
    #
    # print("-----------------------------")
    # print(str(np.mean(pkt_loss_drl_all) * 100) + ' %')
    # print("DRL Throughput " + str(np.mean(rw_drl_all)) + " Mbps")
    # print("Delay " + str(np.mean(pkt_d_all)))
    t_pkt_loss_all.append(np.mean(pkt_loss_all))
    t_pkt_loss_drl_all.append(np.mean(pkt_loss_drl_all))
    t_rw_sh_all.append(np.mean(rw_sh_all))
    t_rw_drl_all.append(np.mean(rw_drl_all))
    t_pkt_d_sch_all.append(np.mean(pkt_d_sch_all))
    t_pkt_d_all.append(np.mean(pkt_d_all))

figure, axis1 = plt.subplots(1)

axis1.plot(x, t_pkt_loss_all, label=env1.schedulers[0].name)
axis1.plot(x, t_pkt_loss_drl_all, label='rDRL-SRA$_{DQN}$')
axis1.set_xlabel('Mean user traffic load (Mbps)')
axis1.set_ylabel('Mean user packet loss (%)')
#axis2.set_title('Packet loss per episode')
axis1.legend(loc=7)

figure2, axis2 = plt.subplots(1)

axis2.plot(x, t_pkt_d_sch_all, label=env1.schedulers[0].name)
axis2.plot(x, t_pkt_d_all, label='rDRL-SRA$_{DQN}$')
axis2.set_xlabel('Mean user traffic load (Mbps)')
axis2.set_ylabel('Mean user packet age')
#axis2.set_title('Packet loss per episode')
axis2.legend(loc=7)

figure3, axis3 = plt.subplots(1)

axis3.plot(x, t_rw_sh_all, label=env1.schedulers[0].name)
axis3.plot(x, t_rw_drl_all, label='rDRL-SRA$_{DQN}$')
axis3.set_xlabel('Mean user traffic load (Mbps)')
axis3.set_ylabel('Mean user throughput (Mbps)')
#axis2.set_title('Packet loss per episode')
axis3.legend(loc=7)

plt.show()
