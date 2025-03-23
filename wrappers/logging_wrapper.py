import os
import gym
import json
import numpy as np

class LoggingWrapper(gym.Wrapper):
    """Trajectory logging wrapper
    
    Records:
    - All observations
    - All actions taken
    - Episode metadata
    
    Persists trajectories to JSON files
    """
    
    def __init__(self, env, folder="trajs", file_id=None):
        super().__init__(env)
        self.trajs = []
        self.traj = {"observations": [], "actions": []}
        self.folder = folder
        self.file_id = (np.random.randint(0, 10000000) 
                       if file_id is None else file_id)
        self.file_path = f"{self.folder}/{self.file_id}.json"
        os.makedirs("trajs", exist_ok=True)

    def __len__(self):
        return len(self.env.data)

    def reset(self, seed=None, return_info=False, options=None, index=None):
        """Reset environment and initialize new trajectory"""
        output = self.env.reset(seed=seed, return_info=return_info, options=options, index=index)
        observation = output[0] if return_info else output
        self.traj = {"observations": [observation], "actions": []}
        return output

    def step(self, action):
        """Record step data and propagate action"""
        obs, reward, done, info = self.env.step(action)
        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)
        if done:
            self.traj.update(info)
        return obs, reward, done, info

    def update_record(self):
        """Archive completed trajectory"""
        if len(self.traj) > 0:
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": []}

    def write(self):
        """Persist trajectories to disk"""
        self.update_record()
        with open(self.file_path, "w") as f:
            json.dump(self.trajs, f, indent=2)
            print(f"Saved trajectories to {self.file_path}")

    def close(self):
        """Cleanup and final write"""
        self.write()