import argparse


# Function for parsing arguments
def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument("-ds", "--data_set", type=str, choices=['FEVER',], default='FEVER')

    # Parse the arguments
    return parser.parse_args()