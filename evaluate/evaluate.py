import time
from react import webthink

def evaluate(indexes = None, prompt = '', to_print=False, env=None, client=None):
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