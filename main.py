import os
import sys
import json
import yaml
import time
import random
import getpass
import numpy as np
from react import llm
from openai import OpenAI
from wikienv import WikiEnv
from utils.argparser import arg_parse
from wrappers.fever_wrapper import FeverWrapper
from wrappers.logging_wrapper import LoggingWrapper
from evaluate.evaluate import eval_qa, eval_alfworld
from wrappers.hotpotqa_wrapper import HotPotQAWrapper
from alfworld.agents.environment import get_environment

if __name__ == "__main__":
    os.environ["ALFWORLD_DATA"] = "/home/cvijo/.cache/alfworld"

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
        eval_qa(indexes=indexes, prompt=prompt, to_print=True, env=env, client=client)
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
        eval_qa(indexes=indexes, prompt=prompt, to_print=True, env=env, client=client)

    elif args.data_set == 'ALFWorld':
        # Load ALFWorld configuration
        with open('base_config.yaml') as reader:
            config = yaml.safe_load(reader)
        
        # Split for ALFWorld
        split = "eval_out_of_distribution"

        # Create ALFWorld enviornment
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)
        
        # Load the prompt file for fewshot prompting
        prompt_file = './prompts/alfworld_3prompts.json'
        with open(prompt_file, 'r') as file:
            prompt_examples = json.load(file)

        # Evaluate
        eval_alfworld(env=env, prompt_examples=prompt_examples, client=client)
        
    elif args.data_set == 'WebShop':
        # Logic for WebShop evaluation
        pass
