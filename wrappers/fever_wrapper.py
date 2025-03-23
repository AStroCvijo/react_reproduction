import gym
import json
import numpy as np
from utils.const import FEVER_SPLIT_FILE
from utils.wrapper_utils import normalize_answer

class FeverWrapper(gym.Wrapper):
    """Wrapper for FEVER (Fact Verification) dataset environment
    
    Features:
    - Loads FEVER dataset splits
    - Manages fact verification episodes
    - Calculates verification accuracy
    """
    
    def __init__(self, env, split):
        super().__init__(env)
        data_path = f"{FEVER_SPLIT_FILE[split]}"
        with open(data_path) as json_file:
            json_list = list(json_file)

        self.data = []
        for json_str in json_list:
            data = json.loads(json_str)
            self.data.append((data["claim"], data["label"]))
            
        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, index=None):
        """Reset environment with new claim"""
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step('')  # Workaround for initial state issues
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        
        self.data_idx = (np.random.randint(len(self.data)) 
                        if index is None else index)
        observation = f"Claim: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        return {
            "steps": self.steps,
            "answer": self.answer,
            "question": self.data[self.data_idx][0],
            "fever_split": self.split
        }

    def get_reward(self, info):
        """Calculate binary accuracy reward"""
        if info['answer'] is not None:
            label = normalize_answer(self.data[self.data_idx][1])
            pred = normalize_answer(info['answer'])
            return int(label == pred)
        return 0

    def step(self, action):
        """Execute environment step with reward calculation"""
        obs, _, done, info = self.env.step(action)
        reward = self.get_reward(info)
        if done:
            # Finalize episode metrics
            obs = f"Episode finished, reward = {reward}\n"
            info.update({
                "gt_answer": self.data[self.data_idx][1],
                "question_idx": self.data_idx
            })
            info.update({'em': reward, 'reward': reward, 'f1': reward})
        return obs, reward, done, info

    def __len__(self):
        return len(self.data)