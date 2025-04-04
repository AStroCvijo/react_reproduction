import os
import json
import yaml
import random
import getpass
from openai import OpenAI
from utils.argparser import arg_parse
from wrappers.wiki_wrapper import WikiEnv
from wrappers.webshop_wrapper import webshopEnv
from wrappers.fever_wrapper import FeverWrapper
from wrappers.logging_wrapper import LoggingWrapper
from wrappers.hotpotqa_wrapper import HotPotQAWrapper
from alfworld.agents.environment import get_environment
from evaluate.evaluate import eval_qa, eval_alfworld, eval_webshop, eval_qa_cot_sc_react, eval_qa_react_cot_sc

if __name__ == "__main__":

    # Get the OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Parse the arguments
    args = arg_parse()

    if args.data_set == "FEVER":
        # Wrappers
        wiki_env = WikiEnv()
        fever_env = FeverWrapper(wiki_env, split="dev")
        env = LoggingWrapper(fever_env)

        # Random shuffle - With seed for reproducability
        indexes = list(range(7405))
        random.Random(233).shuffle(indexes)

        # Evaluate on the FEVER dataset
        if args.prompt_style == "CoT-SC-ReAct":
            # Load the prompt
            prompt_file = "./prompts/fever.json"
            with open(prompt_file, "r") as file:
                prompt_data = json.load(file)
                cot_prompt = prompt_data['cot']
                react_prompt = prompt_data['react']

            # Load the instruction
            instruction_file = "./prompts/instructions/fever_instruct.json"
            with open(instruction_file, "r") as file:
                prompt_data = json.load(file)
                cot_instruction = prompt_data['cot']
                react_instruction = prompt_data['react']

            cot_prompt = cot_instruction + cot_prompt
            react_prompt = react_instruction + react_prompt

            eval_qa_cot_sc_react(
                indexes=indexes,
                cot_prompt=cot_prompt,
                react_prompt=react_prompt,
                to_print=True,
                env=env,
                client=client,
                num_samples=5,
                tempreture=0.7,
            )

        elif args.prompt_style == "ReAct-CoT-SC":
            # Load the prompt
            prompt_file = "./prompts/fever.json"
            with open(prompt_file, "r") as file:
                prompt_data = json.load(file)
                cot_prompt = prompt_data['cot']
                react_prompt = prompt_data['react']

            # Load the instruction
            instruction_file = "./prompts/instructions/fever_instruct.json"
            with open(instruction_file, "r") as file:
                prompt_data = json.load(file)
                cot_instruction = prompt_data['cot']
                react_instruction = prompt_data['react']

            cot_prompt = cot_instruction + cot_prompt
            react_prompt = react_instruction + react_prompt

            eval_qa_react_cot_sc(
                indexes=indexes,
                cot_prompt=cot_prompt,
                react_prompt=react_prompt,
                to_print=True,
                env=env,
                client=client,
                num_samples=21,
                tempreture=0.7,
                react_max_steps=5
            )
        else:
            # Load the prompt
            prompt_file = "./prompts/fever.json"
            with open(prompt_file, "r") as file:
                prompt = json.load(file)[args.prompt_style.lower()]

            # Load the instruction
            instruction_file = "./prompts/instructions/fever_instruct.json"
            with open(instruction_file, "r") as file:
                instruction = json.load(file)[args.prompt_style.lower()]

            prompt = instruction + prompt

            eval_qa(
                indexes=indexes,
                prompt=prompt,
                to_print=True,
                env=env,
                client=client,
                num_samples=args.num_samples,
                tempreture=args.tempreture,
            )

    elif args.data_set == "HotpotQA":
        # Wrappers
        wiki_env = WikiEnv()
        hotpotqa_env = HotPotQAWrapper(wiki_env, split="dev")
        env = LoggingWrapper(hotpotqa_env)

        # Load the prompt
        prompt_file = "./prompts/hotpotqa.json"
        with open(prompt_file, "r") as file:
            prompt = json.load(file)[args.prompt_style.lower()]

        # Load the instruction
        instruction_file = "./prompts/instructions/hotpotqa_instruct.json"
        with open(prompt_file, "r") as file:
            instruction = json.load(file)[args.prompt_style.lower()]

        prompt = instruction + prompt

        # Random shuffle - With seed for reproducability
        indexes = list(range(7405))
        random.Random(233).shuffle(indexes)

        # Evaluate on the HotpotQA dataset
        if args.prompt_style.lower() == "cot-sc":
            eval_qa(
                indexes=indexes,
                prompt=prompt,
                to_print=True,
                env=env,
                client=client,
                num_samples=21,
                tempreture=0.7,
            )
        else:
            eval_qa(
                indexes=indexes,
                prompt=prompt,
                to_print=True,
                env=env,
                client=client,
                num_samples=args.num_samples,
                tempreture=args.tempreture,
            )

    elif args.data_set == "ALFWorld":
        # Load ALFWorld configuration
        with open("config/ALFWorld_config.yaml") as reader:
            config = yaml.safe_load(reader)

        # Split for ALFWorld
        split = "eval_out_of_distribution"

        # Create ALFWorld enviornment
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)

        # Load the prompt file for fewshot prompting
        prompt_file = "./prompts/alfworld.json"
        with open(prompt_file, "r") as file:
            prompt_examples = json.load(file)

        # Evaluate
        eval_alfworld(
            env=env,
            prompt_examples=prompt_examples,
            client=client,
            prompt_style=args.prompt_style.lower(),
        )

    elif args.data_set == "WebShop":
        # Load the prompt
        prompt_file = "./prompts/webshop.json"
        with open(prompt_file, "r") as file:
            prompt = json.load(file)[args.prompt_style.lower()]

        # Create WebShop enviornment
        env = webshopEnv()

        # Evaluate
        eval_webshop(env=env, prompt=prompt, n=50, client=client)
