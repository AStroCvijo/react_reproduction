# -------------------------------------------------------------------------
# OpenAI LLM
# -------------------------------------------------------------------------

def llm(prompt, stop=["\n"], client=None):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
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