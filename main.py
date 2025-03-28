import os
import json
import yaml
import random
import getpass
from react import llm
from openai import OpenAI
from wikienv import WikiEnv
from utils.argparser import arg_parse
from wrappers.webshop_wrapper import webshopEnv
from wrappers.fever_wrapper import FeverWrapper
from wrappers.logging_wrapper import LoggingWrapper
from wrappers.hotpotqa_wrapper import HotPotQAWrapper
from alfworld.agents.environment import get_environment
from evaluate.evaluate import eval_qa, eval_alfworld, eval_webshop

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
            prompt = json.load(file)[args.prompt_style.lower()]

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
        prompt_file='./prompts/hotpotqa.json'
        with open(prompt_file, 'r') as file:
            prompt = json.load(file)[args.prompt_style.lower()]
        
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
        prompt_file = './prompts/alfworld.json'
        with open(prompt_file, 'r') as file:
            prompt_examples = json.load(file)

        # Evaluate
        eval_alfworld(env=env, prompt_examples=prompt_examples, client=client, prompt_style=args.prompt_style.lower())
        
    elif args.data_set == 'WebShop':
        # Load the prompt
        prompt_file = './prompts/webshop.json'
        with open(prompt_file, 'r') as file:
            prompt = json.load(file)[args.prompt_style.lower()]

        # Create WebShop enviornment
        env = webshopEnv()

        # Evaluate
        eval_webshop(env=env, prompt=prompt, n=50, client=client)