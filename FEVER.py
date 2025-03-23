import os
import sys
import json
import time
import random
from openai import OpenAI
from wikienv import WikiEnv
from wrappers.history_wrapper import HistoryWrapper
from wrappers.hotpotqa_wrapper import HotPotQAWrapper
from wrappers.logging_wrapper import LoggingWrapper
from wrappers.fever_wrapper import FeverWrapper

# -------------------------------------------------------------------------
# OpenAI LLM
# -------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Function for generating an llm answer
def llm(prompt, stop=["\n"]):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "system", "content": "You are an assistant that generates helpful answers."},
              {"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=stop)
    return response.choices[0].message.content

# -------------------------------------------------------------------------
# Wrappers
# -------------------------------------------------------------------------

wiki_env = WikiEnv()                               # Wikipedia search class
fever_env = FeverWrapper(wiki_env, split='dev')    # Pass the Wikipedia env to Fever wrapper
env = LoggingWrapper(fever_env)                    # Pass it to the Logging wrapper

# Function for making the next ReAct step
def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# -------------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------------

# Path to fever prompt template
prompt_file = './prompts/fever.json'
with open(prompt_file, 'r') as f:
    prompt_dict = json.load(f)

prompt = prompt_dict['webthink_simple3']

# -------------------------------------------------------------------------
# ReAct
# -------------------------------------------------------------------------

# Function for executing ReAct-style reasoning for fact verification using Wikipedia environment.
def webthink(index=None, prompt=prompt, to_print=True):
    """Execute ReAct-style reasoning for fact verification using Wikipedia environment.
    
    Args:
        index: Index of the question in the dataset
        prompt: Initial prompt template with instructions
        to_print: Whether to print intermediate steps
    
    Returns:
        r: Reward (1 for correct, 0 for incorrect)
        info: Dictionary containing additional information
    """
    # Initialize environment with specific question index
    question = env.reset(index=index)
    if to_print:
        print(index, question)
    
    # Build initial prompt with current question
    prompt += question + '\n'
    
    # Track LLM API call statistics
    n_calls, n_badcalls = 0, 0
    
    # ReAct loop - maximum 20 reasoning steps
    for i in range(1, 20):
        n_calls += 1
        # Generate thought and action using LLM
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        
        try:
            # Attempt to split thought and action components
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except ValueError:
            # Handle malformed responses by making separate calls for thought and action
            if to_print:
                print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            # Get first line as thought
            thought = thought_action.strip().split('\n')[0]
            # Generate action separately if parsing failed
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        
        # Execute action in environment
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        
        # Format step and add to growing prompt
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        
        if to_print:
            print(step_str)
        
        # Early exit if environment indicates completion
        if done:
            break
    
    # Finalize if max steps reached without completion
    if not done:
        obs, r, done, info = step(env, "finish[]")
    
    if to_print:
        print(info, '\n')
    
    # Add debugging information to return dict
    info.update({
        'n_calls': n_calls,          # Total LLM API calls made
        'n_badcalls': n_badcalls,    # Number of malformed responses
        'traj': prompt               # Full trajectory for analysis
    })
    
    return r, info

# -------------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------------

# Random shuffle - With seed for reproducability
indexes = list(range(7405))
random.Random(233).shuffle(indexes)

# Loop over firts 500 questions
answers = []
infos = []
old_time = time.time()
for i in indexes[:500]:
    _, info = webthink(index=i, prompt=prompt, to_print=True)
    answers.append(info['em'])
    infos.append(info)
    print(f'Correct answers: {sum(answers)}, Total questions: {len(answers)}, Accuracy: {(sum(answers) / len(answers))*100}%, Time: {(time.time() - old_time) / len(answers)}')
    print('-------------------------------------------------------------------------------------------------\n')