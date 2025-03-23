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