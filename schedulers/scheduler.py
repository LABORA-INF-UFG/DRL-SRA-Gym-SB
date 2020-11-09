class Scheduler:

    def policy_action(self):
        pass

    def clear(self):
        self.alloc_users = [[] for i in range(len(self.F))]
        #self.pointer = 0

    def reset(self):
        self.buffers.reset_buffers()