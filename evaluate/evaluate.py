import time
from utils.const import prefixes
from react import webthink, alfworld_run

# Evaluation script for FEVER and HotpotQA datasets
def eval_qa(indexes = None, prompt = '', to_print=False, env=None, client=None):
    answers = []
    infos = []
    old_time = time.time()

    # Loop over firts 500 questions
    for i in indexes[:500]:
        _, info = webthink(index=i, prompt=prompt, to_print=True, env=env, client=client)
        answers.append(info['em'])
        infos.append(info)
        print(      
            f"Correct answers: {sum(answers)}, "
            f"Total questions: {len(answers)}, "
            f"Accuracy: {(sum(answers) / len(answers)) * 100}%, "
            f"Time: {(time.time() - old_time) / len(answers)}"
        )
        print('-------------------------------------------------------------------------------------------------\n')

# Evaluation script for ALFWorld dataset
def eval_alfworld(env=None, prompt_examples='', client=None):

    # Success counters: rs[task_type] = successes, cnts[task_type] = attempts
    cnts = [0] * 6
    rs = [0] * 6

    for task_num in range(134):
        # Reset environment for new task
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])  # Clean observation
        
        # Extract task type from path (e.g., 'pick_and_place/living_room')
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(f"\n[Task {task_num+1}/134]")
        print(f"Task ID: {name}")
        
        # Match task to type and select prompt
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                # Construct prompt with few-shot examples
                prompt = 'Interact with a household to solve a task. The game will end on its own when you complete the task. Here are two examples.\n' + prompt_examples[f'react_{v}_1'] + '\nHere is the task.\n'
                # Run task with ReAct
                r = alfworld_run(prompt, ob=ob, client=client, env=env)
                
                # Update success counters
                rs[i] += r
                cnts[i] += 1
                break
        
        # Progress update
        success_rate = sum(rs) / sum(cnts) if sum(cnts) > 0 else 0
        print(f"\n[Progress]")
        print(f"Completed: {task_num+1}/134")
        print(f"Current Success Rate: {success_rate:.2%}")
        print(f"Per-Task Counts: {dict(zip(prefixes.keys(), cnts))}")
        print(f"Per-Task Successes: {dict(zip(prefixes.keys(), rs))}")
        print('-'*40)