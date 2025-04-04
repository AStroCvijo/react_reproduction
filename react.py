import sys
from collections import Counter

# -------------------------------------------------------------------------
# OpenAI LLM
# -------------------------------------------------------------------------


def llm(prompt, stop=["\n"], client=None, temperature=0):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------------
# ReAct-style reasoning for fact verification using Wikipedia environment
# -------------------------------------------------------------------------


def webthink(
    index=None,
    prompt="",
    to_print=True,
    env=None,
    client=None,
    num_samples=1,
    temperature=0.0,
):
    # Initialize environment
    question = env.reset(index=index)
    if to_print:
        print(index, question)

    prompt += question + "\n"

    # Collect multiple sampled answers
    all_answers = []
    n_calls, n_badcalls = 0, 0

    print(num_samples)
    for sample_idx in range(num_samples):
        # Reset prompt for each trajectory
        sample_prompt = prompt

        # ReAct-style reasoning loop
        for i in range(1, 8):
            n_calls += 1
            thought_action = llm(
                sample_prompt + f"Thought {i}:",
                stop=[f"\nObservation {i}:"],
                client=client,
                temperature=temperature,
            )

            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except ValueError:
                if to_print:
                    print("Malformed response:", thought_action)
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split("\n")[0]
                action = llm(
                    sample_prompt + f"Thought {i}: {thought}\nAction {i}:",
                    stop=[f"\n"],
                    client=client,
                ).strip()

            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace("\\n", "")

            # Append step to prompt
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            sample_prompt += step_str

            if to_print:
                print(step_str)

            if done:
                break

        # Ensure we complete the process
        if not done:
            obs, r, done, info = step(env, "finish[]")

        all_answers.append(info.get("answer", "").strip())

    # Perform majority voting and get the most common answer
    if num_samples > 1:
        answer_counts = Counter(all_answers)
        final_answer = answer_counts.most_common(1)[0][0]
    else:
        final_answer = all_answers

    # Update `info` with self-consistent answer and additional details
    info.update(
        {
            "answer": final_answer,  # Use majority-voted answer
            "n_calls": n_calls,  # Total LLM API calls
            "n_badcalls": n_badcalls,  # Number of malformed responses
            "trajectories": all_answers,  # Store all sampled answers
        }
    )

    if to_print:
        print("\nFinal Answer (Majority Vote):", final_answer)

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
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def alfworld_run(prompt, to_print=True, ob="", client=None, env=None):
    init_prompt = prompt + ob + "\n>"
    prompt = ""
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        action = llm(init_prompt + prompt, stop=["\n"], client=client).strip()
        if action.startswith("put"):
            action = action.replace("put", "move", 1)
            for prep in [" in/on ", " in ", " on "]:
                if prep in action:
                    action = action.replace(prep, " to ", 1)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info["won"][0], done[0]
        # print(f"Infos: {info}\n")
        if action.startswith("think:"):
            observation = "OK."
        if to_print:
            print(f"Act {i}: {action}\nObs {i}: {observation}")
            sys.stdout.flush()
        prompt += f" {action}\n{observation}\n>"
        if done:
            return reward
    return 0


# -------------------------------------------------------------------------
# ReAct-style reasoning for WebShop dataset
# -------------------------------------------------------------------------


def webshop_run(idx, prompt, to_print=True, env=None, client=None):
    action = "reset"
    init_prompt = prompt
    prompt = ""
    for i in range(15):
        try:
            res = env.step(idx, action)
            observation = res[0]
        except AssertionError:
            observation = "Invalid action!"

    if action.startswith("think"):
        observation = "OK."

    if to_print:
        print(f"Action: {action}\nObservation: {observation}\n")
        sys.stdout.flush()
    if i:
        prompt += f" {action}\nObservation: {observation}\n\nAction:"
    else:
        prompt += f"{observation}\n\nAction:"

    if res[2]:
        return res[1]

    action = llm(
        init_prompt + prompt[-(6400 - len(init_prompt),) :], stop=["\n"], client=client
    ).lstrip(" ")

    return 0
