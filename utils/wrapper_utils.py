import re
import time
import string
from collections import Counter

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