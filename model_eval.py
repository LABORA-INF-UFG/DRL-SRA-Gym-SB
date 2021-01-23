import numpy as numpy
from stable_baselines import ACKTR, A2C, TRPO, DQN, PPO1, PPO2, ACER
import json
from json import JSONEncoder
from sra_env.sra_env3 import SRAEnv

import consts
import copy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

class ModelEval():

    def __init__(self, model=None):
        self.model = model
        self.env1 = SRAEnv()
        self.env1.running_tp = 1 # validation with different dataset
        self.env1.type = "Master"
        self.env1.reset()
        self.model_name = None
        obs, rw, endep, info  = self.env1.step_(0)
        self.obs = obs
        self.F = "_F_" + consts.F_D + "_ME"
        # number of executions/episodes per trained models
        self.t = 2
        self.simulation_type = "n-stationary"
        #self.folder = '/content/drive/MyDrive/DRL-SRA-Gym/models_final_p/'
        self.folder = 'models_final_p/'
        self.base_file = self.F + "_gamma_" + consts.GAMMA_D + "_lr_" + consts.LR_D + '_epsilon_' + consts.EPSILON_D

    def test(self, episode_history=0):
        print("Testing... using model with " + str(episode_history) + " episodes")
        # saving the model
        self.model.save(self.folder + self.model_name + "_" + str(episode_history) + self.base_file)

        # repository
        rewards_drl_agent_1, rewards_drl_agent_2, rewards_drl_agent_3, rewards_drl_agent_4, rewards_drl_agent_5, \
        rewards_drl_agent_6, rewards_drl_agent_7 = [], [], [], [], [], [], []
        reward_schedulers = []
        pkt_loss_1, pkt_loss_2, pkt_loss_3, pkt_loss_4, pkt_loss_5, pkt_loss_6, pkt_loss_7 = [], [], [], [], [], [], []
        pkt_d_1, pkt_d_2, pkt_d_3, pkt_d_4, pkt_d_5, pkt_d_6, pkt_d_7 = [], [], [], [], [], [], []
        pkt_loss_sch = []
        pkt_delay_sch = []

        # loop for "Monte Carlo" validation
        for ep in range(self.t):
            print("Episode " + str(ep + 1) + " / " + str(self.t))
            p_rewards_drl_agent_1, p_rewards_drl_agent_2, p_rewards_drl_agent_3, p_rewards_drl_agent_4, \
            p_rewards_drl_agent_5, p_rewards_drl_agent_6, p_rewards_drl_agent_7 = [], [], [], [], [], [], []
            p_pkt_loss_1, p_pkt_loss_2, p_pkt_loss_3, p_pkt_loss_4, p_pkt_loss_5, p_pkt_loss_6, p_pkt_loss_7 = [], [], [], [], [], [], []
            p_pkt_d_1, p_pkt_d_2, p_pkt_d_3, p_pkt_d_4, p_pkt_d_5, p_pkt_d_6, p_pkt_d_7 = [], [], [], [], [], [], []
            p_reward_schedulers = []
            p_pkt_loss_sch = []
            p_pkt_delay_sch = []

            # running a full episode
            for i in range(self.env1.blocks_ep):
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, rewards, _, _ = self.env1.step_(action)
                p_rewards_drl_agent_1.append(rewards[0])
                # the last one is the drl agent pkt loss/delay
                p_pkt_loss_1.append(rewards[2][0])
                p_pkt_d_1.append(rewards[3][0])

                rw_sh = [[] for i in range(len(self.env1.schedulers))]
                pkt_l_sh = [[] for i in range(len(self.env1.schedulers))]
                pkt_d_sh = [[] for i in range(len(self.env1.schedulers))]
                ## schedulers
                for u, v in enumerate(self.env1.schedulers):
                    rw_sh[u].append(rewards[1][u])
                    pkt_l_sh[u].append(rewards[2][u + 1])
                    pkt_d_sh[u].append(rewards[3][u + 1])

                p_reward_schedulers.append(rw_sh)
                p_pkt_loss_sch.append(pkt_l_sh)
                p_pkt_delay_sch.append(pkt_d_sh)

            rewards_drl_agent_1.append(p_rewards_drl_agent_1)
            pkt_loss_1.append(p_pkt_loss_1)
            pkt_d_1.append(p_pkt_d_1)
            reward_schedulers.append(p_reward_schedulers)
            pkt_loss_sch.append(p_pkt_loss_sch)
            pkt_delay_sch.append(p_pkt_delay_sch)

        ## saving history data
        history_data = {
            "rewards": {
                "A2C": rewards_drl_agent_1,
                "schedulers": reward_schedulers
            },
            "pkt_loss": {
                "A2C": pkt_loss_1,
                "schedulers": pkt_loss_sch
            },
            "pkt_delay": {
                "A2C": pkt_d_1,
                "schedulers": pkt_delay_sch
            }
        }

        with open('history_final_p/' + self.simulation_type + self.F + '_' + str(self.t)
                  +'_rounds_'+str(self.env1.blocks_ep)+'_bloks_eps_lr_'+ consts.LR_D +'_'+str(episode_history)
                  +'_episodes.json', 'w') as outfile:
            json.dump(history_data, outfile, cls=NumpyArrayEncoder)


    def test__(self, episode_history=0):

        history_data={
            "rewards":{
                "drl": [],
                "schedulers": []
            },
            "pkt_loss":{
                "drl": [],
                "schedulers": []
            },
            "pkt_delay":{
                "drl": [],
                "schedulers": []
            }
        }

        print("Testing... " + str(episode_history) + " episodes")
        self.model.save(self.folder+self.model_name+"_"+str(episode_history)+self.base_file)
        for i in range(0,self.t):
            print("T " + str(i + 1) + " / " + str(self.t))
            action, _ = self.model.predict(self.obs,deterministic=True)
            self.obs, rewards, _, _ = self.env1.step_(action)
            history_data['rewards']['drl'].append(rewards[0])
            # the last one is the drl agent pkt loss/delay
            history_data['pkt_loss']['drl'].append(rewards[2][0])
            history_data['pkt_delay']['drl'].append(rewards[3][0])
            print(action)
            rw_sh = [[] for i in range(len(self.env1.schedulers))]
            pkt_l_sh = [[] for i in range(len(self.env1.schedulers))]
            pkt_d_sh = [[] for i in range(len(self.env1.schedulers))]
            ## schedulers
            for u, v in enumerate(self.env1.schedulers):
                rw_sh[u].append(rewards[1][u])
                pkt_l_sh[u].append(rewards[2][u + 1])
                pkt_d_sh[u].append(rewards[3][u + 1])

            history_data['rewards']['schedulers'].append(rw_sh)
            history_data['pkt_loss']['schedulers'].append(pkt_l_sh)
            history_data['pkt_delay']['schedulers'].append(pkt_d_sh)

        #print(history_data)
        #with open('/content/drive/MyDrive/DRL-SRA-Gym/history_final_p/'+self.simulation_type+ self.F +'_'+str(self.t)
        with open('history_final_p/' + self.simulation_type + self.F + '_' + str(self.t)
                  +'_rounds_'+str(self.env1.blocks_ep)+'_bloks_eps_lr_'+ consts.LR_D +'_'+str(episode_history)
                  +'_episodes.json', 'w') as outfile:
            json.dump(history_data, outfile, cls=NumpyArrayEncoder)