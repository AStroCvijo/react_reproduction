import gym

class HistoryWrapper(gym.ObservationWrapper):
    """Wrapper that formats observations as either current state or full history
    
    Args:
        env: Gym environment to wrap
        obs_format (str): Format mode - 'obs' for current observation, 
                         'history' for full trajectory
        prompt (str, optional): Initial prompt prefix for history format
    """
    
    def __init__(self, env, obs_format, prompt=None):
        super().__init__(env)
        assert obs_format in ["obs", "history"]
        if obs_format == "history":
            assert hasattr(self.env, "traj")
        self.obs_format = obs_format
        self.prompt = prompt if prompt is not None else ""

    def observation(self, obs):
        """Format observation based on specified format"""
        if self.obs_format == "obs":
            return obs
        elif self.obs_format == "history":
            # Build complete interaction history
            observation = self.env.traj["observations"][0] + "\n"
            for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], 
                                        self.env.traj["actions"]), 1):
                observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
            return self.prompt + observation