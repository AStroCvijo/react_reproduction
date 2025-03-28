import argparse


# Function for parsing arguments
def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument("-ds", "--data_set", type=str, choices=['FEVER', 'HotpotQA', 'ALFWorld', 'WebShop'], default='FEVER')
    parser.add_argument("ps", "--prompt_style", type=str, choices=['ReAct', 'Act', 'CoT', 'Standard'], default='ReAct')

    # Parse the arguments
    return parser.parse_args()