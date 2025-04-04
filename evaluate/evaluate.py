import time
from utils.const import prefixes
from react import webthink, alfworld_run, webshop_run, webthink_cot_sc_react, webthink_react_cot_sc


def print_separator(symbol="=", length=100):
    print(symbol * length)


# -------------------------------------------------------------------------
# Evaluation script for FEVER and HotpotQA datasets
# -------------------------------------------------------------------------


def eval_qa(
    indexes=None,
    prompt="",
    to_print=False,
    env=None,
    client=None,
    num_samples=1,
    tempreture=0.0,
):
    answers = []
    infos = []
    start_time = time.time()

    print_separator()
    print(f"🚀 Starting QA Evaluation - {len(indexes[:500])} questions")
    print_separator()

    for idx, i in enumerate(indexes[:500], 1):
        _, info = webthink(
            index=i,
            prompt=prompt,
            to_print=True,
            env=env,
            client=client,
            num_samples=num_samples,
            temperature=tempreture,
        )
        answers.append(info["em"])
        infos.append(info)

        elapsed = time.time() - start_time
        print(f"\n📊 Question {idx:03d}/{len(indexes[:500]):03d}")
        print(f"   Exact Match: {'✅' if info['em'] else '❌'}")
        print(f"   Cumulative: {sum(answers)}/{len(answers)}")
        print(f"   Accuracy: {sum(answers)/len(answers)*100:.2f}%")
        print(f"   Avg Time: {elapsed/len(answers):.2f}s")
        print_separator("-", 50)

def eval_qa_cot_sc_react(
    indexes=None,
    cot_prompt="",
    react_prompt="",
    to_print=False,
    env=None,
    client=None,
    num_samples=1,
    tempreture=0.0,
):
    answers = []
    infos = []
    start_time = time.time()

    print_separator()
    print(f"🚀 Starting QA Evaluation - {len(indexes[:500])} questions")
    print_separator()

    for idx, i in enumerate(indexes[:500], 1):
        _, info = webthink_cot_sc_react(
            index=i,
            cot_prompt=cot_prompt,
            react_prompt=react_prompt,
            to_print=True,
            env=env,
            client=client,
            num_samples=num_samples,
            temperature=tempreture,
        )
        answers.append(info["em"])
        infos.append(info)

        elapsed = time.time() - start_time
        print(f"\n📊 Question {idx:03d}/{len(indexes[:500]):03d}")
        print(f"   Exact Match: {'✅' if info['em'] else '❌'}")
        print(f"   Cumulative: {sum(answers)}/{len(answers)}")
        print(f"   Accuracy: {sum(answers)/len(answers)*100:.2f}%")
        print(f"   Avg Time: {elapsed/len(answers):.2f}s")
        print_separator("-", 50)

def eval_qa_react_cot_sc(
    indexes=None,
    cot_prompt="",
    react_prompt="",
    to_print=False,
    env=None,
    client=None,
    num_samples=1,
    tempreture=0.0,
    react_max_steps=1
):
    answers = []
    infos = []
    start_time = time.time()

    print_separator()
    print(f"🚀 Starting QA Evaluation - {len(indexes[:500])} questions")
    print_separator()

    for idx, i in enumerate(indexes[:500], 1):
        _, info = webthink_react_cot_sc(
            index=i,
            cot_prompt=cot_prompt,
            react_prompt=react_prompt,
            to_print=True,
            env=env,
            client=client,
            num_samples=num_samples,
            temperature=tempreture,
            react_max_steps=react_max_steps
        )
        answers.append(info["em"])
        infos.append(info)

        elapsed = time.time() - start_time
        print(f"\n📊 Question {idx:03d}/{len(indexes[:500]):03d}")
        print(f"   Exact Match: {'✅' if info['em'] else '❌'}")
        print(f"   Cumulative: {sum(answers)}/{len(answers)}")
        print(f"   Accuracy: {sum(answers)/len(answers)*100:.2f}%")
        print(f"   Avg Time: {elapsed/len(answers):.2f}s")
        print_separator("-", 50)

# -------------------------------------------------------------------------
# Evaluation script for ALFWorld dataset
# -------------------------------------------------------------------------


def eval_alfworld(env=None, prompt_examples="", client=None, prompt_style="react"):
    tasks = [0] * 6
    completed = [0] * 6

    print_separator()
    print("🏠 Starting ALFWorld Evaluation - 134 tasks")
    print_separator()

    for task_num in range(134):
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        print(f"\n🔧 Task {task_num+1:03d}/134")
        print(f"   Type: {name}")
        print_separator("-", 50)

        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                prompt = (
                    "Interact with a household to solve a task. The game will end on its own when you complete the task. Here are two examples.\n"
                    + prompt_examples[f"{prompt_style.lower()}_{v}_1"]
                    + "\nHere is the task.\n"
                )
                r = alfworld_run(prompt, ob=ob, client=client, env=env)

                completed[i] += r
                tasks[i] += 1
                break

        success_rate = sum(completed) / sum(tasks) if sum(tasks) > 0 else 0
        print("\n📈 Progress Update")
        print(f"   Completed: {task_num+1}/134")
        print(f"   Success Rate: {success_rate:.2%}")
        print("\n🔍 Task Breakdown:")
        for (task_name), attempts, successes in zip(prefixes.keys(), tasks, completed):
            if attempts > 0:
                print(
                    f"   {task_name:<20}: {successes}/{attempts} ({successes/attempts:.2%})"
                )
        print_separator()


# -------------------------------------------------------------------------
# Evaluation script for WebShop dataset
# -------------------------------------------------------------------------


def eval_webshop(env=None, prompt="", n=50, client=None):
    completed = []
    tasks = 0

    print_separator()
    print(f"🛍️  Starting WebShop Evaluation - {n} trials")
    print_separator()

    for i in range(n):
        print(f"\n🔄 Trial {i+1:02d}/{n}")
        try:
            r = webshop_run(f"fixed_{i}", prompt, to_print=True, env=env, client=client)
        except AssertionError:
            r = 0
            tasks += 1
        completed.append(r)

        if (i + 1) % 5 == 0 or i == n - 1:
            success_count = sum(completed)
            avg_reward = sum(completed) / len(completed)
            success_rate = success_count / len(completed)
            failure_rate = tasks / len(completed)

            print("\n📊 Interim Results")
            print(f"   Trials Completed: {i+1:02d}/{n}")
            print(f"   Average Reward:   {avg_reward:.2f}")
            print(f"   Success Rate:     {success_rate:.2%}")
            print(f"   Failure Rate:     {failure_rate:.2%}")
            print_separator("-", 40)

    print("\n🎯 Final Results")
    print(f"   Total Trials:      {n}")
    print(f"   Average Reward:    {sum(completed)/n:.2f}")
    print(f"   Total Successes:   {sum(completed)}/{n} ({sum(completed)/n:.2%})")
    print(f"   Total Failures:    {tasks}/{n} ({tasks/n:.2%})")
    print_separator()
