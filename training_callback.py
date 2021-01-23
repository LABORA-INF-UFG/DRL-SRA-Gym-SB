from stable_baselines.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, eval_interval=10, episode_length=100, model_eval=None, name=None):
        super(TrainingCallback, self).__init__(verbose)
        self.step_count = 0
        self.episode_count = 0
        self.eval_interval = eval_interval
        self.episode_length = episode_length
        self.model_eval = model_eval
        self.episode_history = 0     
        self.name = name   

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
        self.step_count += 1
        #print("step_count=" + str(self.step_count))
        # counting episodes
        if self.step_count == self.episode_length:
          self.episode_count += 1
          self.episode_history += 1
          self.step_count = 0

        # controlling model evaluation trigger
        if self.episode_count == self.eval_interval:
          self.episode_count = 0
          self.model_eval.model = self.model
          self.model_eval.model_name = self.name
          # running the model evaluation
          self.model_eval.test(episode_history=self.episode_history)
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