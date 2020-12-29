import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as mlpp
from stable_baselines import A2C, ACKTR, TRPO, ACER, DQN, PPO1, PPO2
from sra_env.sra_env2 import SRAEnv
from tqdm import tqdm
import consts

## training models using combinatorial action space

env = SRAEnv()

model1 = A2C(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR, epsilon=consts.EPSILON)
model2 = ACKTR(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
model3 = TRPO(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, vf_stepsize=consts.LR)
model4 = ACER(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
model5 = DQN(mlpp, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
model6 = PPO1(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, adam_epsilon=consts.EPSILON)
model7 = PPO2(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
tqdm_e = tqdm(range(0,220,5), desc='Time Steps', leave=True, unit=" time steps")
#folder = consts.MODELS_FOLDER_STATIONARY
folder = consts.MODELS_FOLDER
F = "_F_3-3_NR_ND" # NR - No repetition - combinatorial action space ND - New Data
for i in tqdm_e:
    #i = 1 if i == 0 else i
    #ts = consts.BLOCKS_EP * i
    ts = consts.BLOCKS_EP * (i + 1)
    base_file = F + "_gamma_"+consts.GAMMA_D+"_lr_"+consts.LR_D+'_epsilon_'+consts.EPSILON_D

    model1.learn(total_timesteps=ts) # time steps
    model1.save(folder+"a2c_drl_"+str(ts)+base_file)

    model2.learn(total_timesteps=ts)
    model2.save(folder+"acktr_"+str(ts)+base_file)

    model3.learn(total_timesteps=ts)
    model3.save(folder+"trpo_"+str(ts)+base_file)

    model4.learn(total_timesteps=ts)
    model4.save(folder+"acer_"+str(ts)+base_file)

    model5.learn(total_timesteps=ts)
    model5.save(folder+"dqn_" + str(ts)+base_file)

    model6.learn(total_timesteps=ts)
    model6.save(folder+"ppo1_" + str(ts)+base_file)

    model7.learn(total_timesteps=ts)
    model7.save(folder+"ppo2_" + str(ts)+base_file)
