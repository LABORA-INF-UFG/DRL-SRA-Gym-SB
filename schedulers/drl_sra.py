import gym
import numpy as np

from akpy.MassiveMIMOSystem5 import MassiveMIMOSystem
from akpy.buffers_at_BS import Buffers
from schedulers.scheduler import Scheduler

import consts

class DRLSRA(Scheduler):

    def __init__(self, buffers: Buffers):
        self.K = consts.K
        self.F = consts.F
        self.pointer = self.K-1
        self.alloc_users = [[] for i in range(len(self.F))]
        self.buffers = buffers
        self.observation_space = None
        self.model = None

    def policy_action(self) -> int:
        action, _states = self.model.predict(self.observation_space)
        return action

    def reset(self):
        # Buffer
        self.buffers.reset_buffers()  # Reseting buffers


    def set_model(self, model):
        self.model = model

    def set_observation_space(self,observation_space):
        self.observation_space = observation_space