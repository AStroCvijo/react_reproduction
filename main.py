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
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as env_mod
import yaml
from react import llm

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
        with open('base_config.yaml') as reader:
            config = yaml.safe_load(reader)
            
        split = "eval_out_of_distribution"

        env = env_mod.get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)

        def process_ob(ob):
            if ob.startswith('You arrive at loc '):
                ob = ob[ob.find('. ')+2:]    
            return ob
        
        folder = './prompts/'
        prompt_file = 'alfworld_3prompts.json'
        with open(folder + prompt_file, 'r') as f:
            d = json.load(f)


        def alfworld_run(prompt, to_print=True, ob=''):
            init_prompt = prompt + ob + '\n>'
            prompt = ''
            if to_print:
                print(ob)
                sys.stdout.flush()
            for i in range(1, 50):
                action = llm(init_prompt + prompt, stop=['\n'], client=client).strip()
                observation, reward, done, info = env.step([action])
                observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
                if action.startswith('think:'):
                    observation = 'OK.'
                if to_print:
                    print(f'Act {i}: {action}\nObs {i}: {observation}')
                    sys.stdout.flush()
                prompt += f' {action}\n{observation}\n>'
                if done:
                    return reward
            return 0

        prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }
        cnts = [0] * 6
        rs = [0] * 6

        for _ in range(134):
            ob, info = env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            
            print(f"Game Name: {name}")
            
            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    prompt = (
                        'Interact with a household to solve a task. Here are two examples.\n'
                        + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'
                    )
                    print(f"Prefix: {k}, Value: {v}")
                    r = alfworld_run(prompt, ob=ob)
                    rs[i] += r
                    cnts[i] += 1
                    break
            
            print(f"Iteration: {_+1}, Reward: {r}, RS: {rs}, Cnts: {cnts}, Average Reward: {sum(rs) / sum(cnts)}")
            print('------------\n')
        
    elif args.data_set == 'WebShop':
        # Logic for WebShop evaluation
        pass
