import sys

# -------------------------------------------------------------------------
# OpenAI LLM
# -------------------------------------------------------------------------

def llm(prompt, stop=["\n"], client=None):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------------------------------
# ReAct-style reasoning for fact verification using Wikipedia environment
# -------------------------------------------------------------------------

def webthink(index=None, prompt='', to_print=True, env = None, client=None):
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
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], client=client)
        
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
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"], client=client).strip()
        
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

# Function for making the next ReAct step
def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# -------------------------------------------------------------------------
# ReAct-style reasoning for ALFWorld
# -------------------------------------------------------------------------

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def alfworld_run(prompt, to_print=True, ob='', client=None, env=None):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        action = llm(init_prompt + prompt, stop=['\n'], client=client).strip()
        if action.startswith('put'):
            action = action.replace('put', 'move', 1)  # Replace "put" with "move"
            # Replace common prepositions ("in", "on", "in/on") with "to"
            for prep in [' in/on ', ' in ', ' on ']:
                if prep in action:
                    action = action.replace(prep, ' to ', 1)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        print(f"Infos: {info}\n")
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if done:
            return reward
    return 0

# -------------------------------------------------------------------------
# ReAct-style reasoning for ALFWorld
# -------------------------------------------------------------------------

def webshop_run(idx, prompt, to_print=True, env=None):
  action = 'reset'
  init_prompt = prompt
  prompt = ''
  for i in range(15):
    try:
      res = env.step(idx, action)
      observation = res[0]
    except AssertionError:
      observation = 'Invalid action!'

    if action.startswith('think'):
      observation = 'OK.'


    if to_print:
      print(f'Action: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {action}\nObservation: {observation}\n\nAction:'
    else:
      prompt += f'{observation}\n\nAction:'
    
    if res[2]:  
      return res[1]

    action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ')

  return 0

def run_episodes(prompt, n=50, env=None):
  rs = []
  cnt = 0
  for i in range(n):
    print('-----------------')
    print(i)
    try:
      r = webshop_run(f'fixed_{i}', prompt, to_print=True, env=env)
    except AssertionError:
      r = 0
      cnt += 1
    rs.append(r)
    if (i+1) % 1 == 0:
      r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
      print(i+1, r, sr, fr)
      print('-------------')
  r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
  print(r, sr, fr)
  return rs