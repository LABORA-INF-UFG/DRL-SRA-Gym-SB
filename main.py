import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C, ACKTR
from sra_env.sra_env import SRAEnv
from tqdm import tqdm
import consts

env = SRAEnv()

model1 = A2C(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR, epsilon=consts.EPSILON)
model2 = ACKTR(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
tqdm_e = tqdm(range(90,5000,5), desc='Time Steps', leave=True, unit=" time steps")
for i in tqdm_e:

    ts = consts.BLOCKS_EP * (i + 1)
    model1.learn(total_timesteps=ts) # time steps
    model1.save("models/a2c_drl_"+str(ts))

    model2.learn(total_timesteps=ts)
    model2.save("models/acktr_"+str(ts))
