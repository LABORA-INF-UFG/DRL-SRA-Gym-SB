import os
import sys

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
from sra_env.sra_env6 import SRAEnv
from utils.tools import NumpyArrayEncoder

env = SRAEnv(type="Master", running_tp=1, desc="DQN")

alpha = list(range(100,101,50))
tqdm_e = tqdm(alpha, desc='Time Steps', leave=True, unit=" time steps")

tss = []

rw_drl_all, pkt_loss_drl_all, pkt_d_drl_all = [], [], []
rw_sch_all = [[] for i in range(len(env.schedulers))]
pkt_loss_sch_all = [[] for i in range(len(env.schedulers))]
pkt_d_sch_all = [[] for i in range(len(env.schedulers))]

for a in tqdm_e:


    b_rw_drl, b_pkt_loss_drl, b_pkt_d_drl = [], [], []
    b_rw_sch = [[] for i in range(len(env.schedulers))]
    b_pkt_loss_sch = [[] for i in range(len(env.schedulers))]
    b_pkt_d_sch = [[] for i in range(len(env.schedulers))]

    for beta in range(0,101,50):

        tss.append([a, beta])

        model = DQN.load('E:/Docs_Doutorado/multi_alfa_beta_nrw/dqn_50000_F_2-2_all_ti11_a'+str(a)+'_b'+str(beta)+'_gamma_07_lr_007_epsilon_1e-05')
        env.reset()
        obs, rw, endep, info = env.step_(0)
        env.reset()

        ep_rw_drl, ep_pkt_loss_drl, ep_pkt_d_drl = [], [], []
        ep_rw_sch = [[] for i in range(len(env.schedulers))]
        ep_pkt_loss_sch = [[] for i in range(len(env.schedulers))]
        ep_pkt_d_sch = [[] for i in range(len(env.schedulers))]

        # running # episodes
        for i in range(5):

            rw_drl, pkt_loss_drl, pkt_d_drl = [], [], []
            rw_sch = [[] for i in range(len(env.schedulers))]
            pkt_loss_sch = [[] for i in range(len(env.schedulers))]
            pkt_d_sch = [[] for i in range(len(env.schedulers))]

            # running a entire episode
            for ii in range(env.blocks_ep):
                action1, _ = model.predict(obs, deterministic=True)
                obs, rewards_1, _, _ = env.step_(action1)
                rw_drl.append(rewards_1[0])
                if rewards_1[2][0] > -10.:
                    pkt_loss_drl.append(rewards_1[2][0])
                pkt_d_drl.append(np.mean(rewards_1[3][0]))
                ## running for each scheduler
                for u, v in enumerate(env.schedulers):
                    if rewards_1[2][u + 1] > -10.:
                        pkt_loss_sch[u].append(rewards_1[2][u + 1])
                    rw_sch[u].append(rewards_1[1][u])
                    pkt_d_sch[u].append(np.mean(rewards_1[3][u + 1]))

            #########################acumulating the episode results

            ep_rw_drl.append(np.mean(rw_drl) / env.K)
            ep_pkt_loss_drl.append(np.mean(pkt_loss_drl))
            ep_pkt_d_drl.append(np.mean(pkt_d_drl))

            for u, v in enumerate(env.schedulers):
                ep_rw_sch[u].append(np.mean(rw_sch[u]) / env.K)
                ep_pkt_loss_sch[u].append(np.mean(pkt_loss_sch[u]))
                ep_pkt_d_sch[u].append(np.mean(pkt_d_sch[u]))

            #########################

        #########################acumulating the beta model results
        b_rw_drl.append(np.mean(ep_rw_drl))
        b_pkt_loss_drl.append(np.mean(ep_pkt_loss_drl))
        b_pkt_d_drl.append(np.mean(ep_pkt_d_drl))

        for u, v in enumerate(env.schedulers):
            b_rw_sch[u].append(np.mean(ep_rw_sch[u]))
            b_pkt_loss_sch[u].append(np.mean(ep_pkt_loss_sch[u]))
            b_pkt_d_sch[u].append(np.mean(ep_pkt_d_sch[u]))
        #########################

    rw_drl_all.append(b_rw_drl)
    pkt_loss_drl_all.append(b_pkt_loss_drl)
    pkt_d_drl_all.append(b_pkt_d_drl)
    for u, v in enumerate(env.schedulers):
        rw_sch_all[u].append(b_rw_sch[u])
        pkt_loss_sch_all[u].append(b_pkt_loss_sch[u])
        pkt_d_sch_all[u].append(b_pkt_d_sch[u])

## savind results
history={
    "rewards_drl_agents":{
        "DQN": rw_drl_all
    },
    "rewards_schedulers": rw_sch_all,
    "pkt_loss_agents":{
        "DQN": pkt_loss_drl_all
    },
    "pkt_loss_schedulers": pkt_loss_sch_all,
    "pkt_delay_schedulers": pkt_d_sch_all,
    "pkt_delay_agents":{
        "DQN": pkt_d_drl_all
    },
    "tss": tss
}

with open('history_multi_a_b/experiment_nrw_1.json', 'w') as outfile:
    json.dump(history, outfile, cls=NumpyArrayEncoder)
#############################