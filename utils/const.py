# Directory configuration for dataset files
DATA_DIR = "data"

# File mappings for different dataset splits
HOTPOTQA_SPLIT_FILE = {
    "train": "data/hotpot_train_v1.1_simplified.json",
    "dev": "data/hotpot_dev_v1_simplified.json",
    "test": "data/hotpot_test_v1_simplified.json",
}

# Fever split file
FEVER_SPLIT_FILE = {
    "train": "data/train.jsonl",
    "dev": "data/paper_dev.jsonl",
}

# ALFWorld constants
prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

# WebShop constants
WEBSHOP_URL = "http://3.83.245.205:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}