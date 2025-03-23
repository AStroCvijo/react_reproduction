import os
import sys
import json
import time
import random
from openai import OpenAI
from wikienv import WikiEnv
from utils.argparser import arg_parse
from evaluate.evaluate import evaluate
from wrappers.fever_wrapper import FeverWrapper
from wrappers.history_wrapper import HistoryWrapper
from wrappers.logging_wrapper import LoggingWrapper
from wrappers.hotpotqa_wrapper import HotPotQAWrapper

if __name__ == "__main__":
    # OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Parse the arguments
    args = arg_parse()
    
    if args.data_set == 'FEVER':
        # Wrappers
        wiki_env = WikiEnv()
        fever_env = FeverWrapper(wiki_env, split='dev')
        env = LoggingWrapper(fever_env)

        # Load the prompt
        prompt_file = './prompts/fever.json'
        with open(prompt_file, 'r') as file:
            prompt = json.load(file)['webthink_simple3']

        # Random shuffle - With seed for reproducability
        indexes = list(range(7405))
        random.Random(233).shuffle(indexes)

        # Evaluate on the FEVER dataset
        evaluate(indexes=indexes, prompt=prompt, to_print=True, env=env, client=client)
    elif args.data_set == 'HotpotQA':
        pass