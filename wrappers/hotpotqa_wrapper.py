import gym
import json
import numpy as np
from utils.const import HOTPOTQA_SPLIT_FILE
from utils.wrapper_utils import normalize_answer, f1_score


class HotPotQAWrapper(gym.Wrapper):
    """Wrapper for HotPotQA dataset environment

    Features:
    - Loads HotPotQA dataset splits
    - Manages question answering episodes
    - Calculates EM and F1 scores
    """

    def __init__(self, env, split):
        super().__init__(env)
        data_file = f"{HOTPOTQA_SPLIT_FILE[split]}"
        with open(data_file) as f:
            self.data = json.load(f)
        self.data = [(d["question"], d["answer"]) for d in self.data]
        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, index=None):
        """Reset environment with new question"""
        # Environment initialization boilerplate
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step("")  # Workaround for initial state issues
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)

        # Set up new question
        self.data_idx = np.random.randint(len(self.data)) if index is None else index
        observation = f"Question: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        """Get current episode metadata"""
        return {
            "steps": self.steps,
            "answer": self.answer,
            "question": self.data[self.data_idx][0],
            "hotpot_split": self.split,
        }

    def get_reward(self, info):
        """Calculate binary exact match reward"""
        if info["answer"] is not None:
            pred = normalize_answer(self.data[self.data_idx][1])
            gt = normalize_answer(info["answer"])
            return int(pred == gt)
        return 0

    def get_metrics(self, info):
        """Calculate full evaluation metrics"""
        if info["answer"] is not None:
            pred = normalize_answer(self.data[self.data_idx][1])
            gt = normalize_answer(info["answer"])
            em = pred == gt
            f1 = f1_score(pred, gt)[0]
            return {"reward": em, "em": em, "f1": f1}
        return {"reward": 0, "em": 0, "f1": 0}

    def step(self, action):
        """Execute environment step with reward calculation"""
        obs, _, done, info = self.env.step(action)
        reward = self.get_reward(info)
        if done:
            # Finalize episode metrics
            obs = f"Episode finished, reward = {reward}\n"
            info.update(
                {
                    "gt_answer": self.data[self.data_idx][1],
                    "question_idx": self.data_idx,
                }
            )
            info.update(self.get_metrics(info))
        return obs, reward, done, info

    def __len__(self):
        return len(self.data)
