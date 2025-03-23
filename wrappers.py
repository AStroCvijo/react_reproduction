import os
import re
import gym
import json
import string
import numpy as np
from collections import Counter

# Directory configuration for dataset files
DATA_DIR = "data"

# File mappings for different dataset splits
HOTPOTQA_SPLIT_FILE = {
    "train": "data/hotpot_train_v1.1_simplified.json",
    "dev": "data/hotpot_dev_v1_simplified.json",
    "test": "data/hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
    "train": "data/train.jsonl",
    "dev": "data/paper_dev.jsonl",
}

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

def normalize_answer(s):
    """Normalize text for answer comparison
    
    Applies:
    1. Lowercasing
    2. Punctuation removal
    3. Article removal (a, an, the)
    4. Whitespace normalization
    
    Args:
        s (str): Input string to normalize
        
    Returns:
        str: Normalized text
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth
    
    Handles special cases for yes/no questions and normalizes inputs
    
    Returns:
        tuple: (f1, precision, recall) scores
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Handle special answer types
    special_answers = ['yes', 'no', 'noanswer']
    if (normalized_prediction in special_answers and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in special_answers and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    
    # Token-based comparison
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
        
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return f1, precision, recall

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
        self.data = [(d['question'], d['answer']) for d in self.data]
        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """Reset environment with new question"""
        # Environment initialization boilerplate
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step('')  # Workaround for initial state issues
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        
        # Set up new question
        self.data_idx = (np.random.randint(len(self.data)) 
                        if idx is None else idx)
        observation = f"Question: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        """Get current episode metadata"""
        return {
            "steps": self.steps,
            "answer": self.answer,
            "question": self.data[self.data_idx][0],
            "hotpot_split": self.split
        }

    def get_reward(self, info):
        """Calculate binary exact match reward"""
        if info['answer'] is not None:
            pred = normalize_answer(self.data[self.data_idx][1])
            gt = normalize_answer(info['answer'])
            return int(pred == gt)
        return 0

    def get_metrics(self, info):
        """Calculate full evaluation metrics"""
        if info['answer'] is not None:
            pred = normalize_answer(self.data[self.data_idx][1])
            gt = normalize_answer(info['answer'])
            em = (pred == gt)
            f1 = f1_score(pred, gt)[0]
            return {'reward': em, 'em': em, 'f1': f1}
        return {'reward': 0, 'em': 0, 'f1': 0}

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
            info.update(self.get_metrics(info))
        return obs, reward, done, info

    def __len__(self):
        return len(self.data)

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

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """Reset environment with new claim"""
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step('')  # Workaround for initial state issues
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        
        self.data_idx = (np.random.randint(len(self.data)) 
                        if idx is None else idx)
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

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        """Reset environment and initialize new trajectory"""
        output = self.env.reset(seed=seed, return_info=return_info, 
                               options=options, idx=idx)
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