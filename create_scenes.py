"""
Load configuration files and generate different scenes for testing
"""
import sys, re, statistics as stat, numpy as np, math, argparse, os, pandas
#from tabulate import tabulate as tab
from copy import deepcopy

def genScenes(input_file_name, executeable_file):
    if not os.path.isfile(input_file_name):
        print('File ', input_file_name, ' holding scene configuration file names does not exist')
        return
    if not os.path.isfile(executeable_file):
        print('Executable ', executeable_file, ' for generating scenes does not exist')
        return
    elif not os.access(executeable_file,os.X_OK):
        print('Unable to execute ', executeable_file)
        return

    cf = pandas.read_csv(input_file_name, delimiter=';')


def main():
    if len(sys.argv) != 2:
        print('Too less or too many arguments')
    else:
        if type(sys.argv[1]) is not str:
            print("First parameter must be a string")
        if type(sys.argv[2]) is not str:
            print("Second parameter must be a string")
        else:
            genScenes(sys.argv[1], sys.argv[2])

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    main()


