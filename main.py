import os
import sys
import json
import time
import random
import getpass
from openai import OpenAI
from wikienv import WikiEnv
from utils.argparser import arg_parse
from evaluate.evaluate import evaluate
from wrappers.fever_wrapper import FeverWrapper
from wrappers.history_wrapper import HistoryWrapper
from wrappers.logging_wrapper import LoggingWrapper
from wrappers.hotpotqa_wrapper import HotPotQAWrapper

if __name__ == "__main__":
    # Get the OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

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
            prompt = json.load(file)['webact_simple3']

        # Random shuffle - With seed for reproducability
        indexes = list(range(7405))
        random.Random(233).shuffle(indexes)

        # Evaluate on the FEVER dataset
        evaluate(indexes=indexes, prompt=prompt, to_print=True, env=env, client=client)
    elif args.data_set == 'HotpotQA':
        # Wrappers
        wiki_env = WikiEnv()
        hotpotqa_env = HotPotQAWrapper(wiki_env, split='dev')
        env = LoggingWrapper(hotpotqa_env)

        # Load the prompt
        prompt_file='./prompts/prompts_naive.json'
        with open(prompt_file, 'r') as file:
            prompt = json.load(file)['webthink_simple6']
        
        instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task.
        Here are some examples.
        """

        prompt = instruction + prompt

        # Random shuffle - With seed for reproducability
        indexes = list(range(7405))
        random.Random(233).shuffle(indexes)

        # Evaluate
        evaluate(indexes=indexes, prompt=prompt, to_print=True, env=env, client=client)
    elif args.data_set == 'ALFWorld':
        # Logic for ALFWorld evaluation
        pass
    elif args.data_set == 'WebShop':
        # Logic for WebShop evaluation
        pass
