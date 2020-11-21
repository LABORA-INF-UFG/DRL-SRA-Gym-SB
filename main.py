import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as mlpp
from stable_baselines import A2C, ACKTR, TRPO, ACER, DQN, PPO1, PPO2
from sra_env.sra_env import SRAEnv
from tqdm import tqdm
import consts

env = SRAEnv()

model1 = A2C(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR, epsilon=consts.EPSILON)
model2 = ACKTR(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
model3 = TRPO(MlpPolicy, env, verbose=0, gamma=consts.GAMMA)
model4 = ACER(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
model5 = DQN(mlpp, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
model6 = PPO1(MlpPolicy, env, verbose=0, gamma=consts.GAMMA)
model7 = PPO2(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
tqdm_e = tqdm(range(0,200,5), desc='Time Steps', leave=True, unit=" time steps")
for i in tqdm_e:
    #i = 1 if i == 0 else i
    #ts = consts.BLOCKS_EP * i
    ts = consts.BLOCKS_EP * (i + 1)
    #model1.learn(total_timesteps=ts) # time steps
    #model1.save("models/a2c_drl_"+str(ts))

    #model2.learn(total_timesteps=ts)
    #model2.save("models/acktr_"+str(ts))
    model3.learn(total_timesteps=ts)
    model3.save("models/trpo_"+str(ts))

    #model4.learn(total_timesteps=ts)
    #model4.save("models/acer_"+str(ts))

    model5.learn(total_timesteps=ts)
    model5.save("models/dqn_" + str(ts))

    model6.learn(total_timesteps=ts)
    model6.save("models/ppo1_" + str(ts))

    model7.learn(total_timesteps=ts)
    model7.save("models/ppo2_" + str(ts))
