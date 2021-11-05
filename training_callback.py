from stable_baselines.common.callbacks import BaseCallback
import numpy as np

class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, eval_interval=10, episode_length=100, env_eval=None, model_name=None, folder=None, save=False):
        super(TrainingCallback, self).__init__(verbose)
        self.total_count = 0
        self.step_count = 0
        self.eval_interval = eval_interval
        self.episode_length = episode_length
        self.env_eval = env_eval
        self.env_eval.set_seed(10)
        self.ep_tests = 10
        self.loss = []
        self.rw = []
        self.delay = []
        #convergence delta
        self.delta = 2
        self.threshold = 1e-2 # for convergence
        self.threshold_count = 0
        self.model_name = model_name
        self.folder = folder
        self.ep_count = 1
        self.save_model=save
        self.error = []


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        self.total_count += 1
        self.step_count += 1
        if self.step_count == self.eval_interval:
            pkt_loss_drl_all, rw_drl_all, pkt_d_all = [], [], []
            actions = []
            episodes = []
            # running for self.ep_tests episodes
            for i in range(self.ep_tests):
                self.env_eval.reset()
                episodes.append(self.env_eval.episode_number)
                obs, rw, endep, info = self.env_eval.step_(0)
                self.env_eval.reset_pre()
                rw_drl = []
                pkt_loss_drl = []
                pkt_d_drl = []
                # running a entire episode
                for ii in range(self.env_eval.blocks_ep):
                    action1, _ = self.model.predict(obs, deterministic=True)
                    actions.append(action1)
                    obs, rewards_1, _, _ = self.env_eval.step_(action1)
                    #print(action1)
                    rw_drl.append(rewards_1[0])
                    if rewards_1[2][0] > -10.:
                        pkt_loss_drl.append(rewards_1[2][0])
                    pkt_d_drl.append(np.mean(rewards_1[3][0]))

                pkt_loss_drl_all.append(np.mean(pkt_loss_drl))
                rw_drl_all.append(np.mean(rw_drl) / self.env_eval.K)
                pkt_d_all.append(pkt_d_drl)

            # accumulating for history
            self.loss.append(np.mean(pkt_loss_drl_all))
            self.rw.append(np.mean(rw_drl_all))
            self.delay.append(np.mean(pkt_d_all))
            #reseting the step counter
            self.step_count = 0
            self.compute_allocations(actions)
            print(episodes)
            #evaluating the convergence
            if len(self.rw) > 10:
                # for rate
                error = np.sqrt(np.power(np.mean(self.rw[-10:-6]) - np.mean(self.rw[-5:-1]),2))
                self.error.append(error)
                print("Error " + str(error))
                if error <= self.threshold and error > 0.0 and error > 7.105427357601002e-15:
                    self.threshold_count += 1
                else:
                    self.threshold_count = 0
                if self.threshold_count > 2:
                    print("Convergence!")
                    print(self.total_count)
                    self.model.save(self.folder + self.model_name + "_" + str(self.env_eval.ep_count))
                    return False

        if self.total_count % 1000 == 0:
            self.ep_count += 1
            if self.ep_count % 10 == 0 and self.save_model and self.ep_count > 40:
                self.save()
                #self.ep_count = 1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def compute_allocations(self, actions):
        allocations = [0 for i in range(self.env_eval.K)]
        for act in actions:
            f_ues = self.env_eval.actions[act]
            for f in f_ues:
                for ue in f:
                    allocations[ue] += 1

        print(allocations)

    def save(self):
        self.model.save(self.folder + self.model_name + "_" + str(self.ep_count) + "_eps")