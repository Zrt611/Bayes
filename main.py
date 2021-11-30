from Data_processing import Generate_Data
from GCNModel import *
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description = 'Configuration')
arglist = []

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arglist.append(arg)
    return arg

## DATA PARAM
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_name',type = str, default = 'segmentation', help = 'use which data to experiment')
data_arg.add_argument('--glass_path',type = str, default = './data/glass/glass.data', help = 'path of glass data')
data_arg.add_argument('--candidate_set_length',type = int, default = 3, help = 'length of candidate set')
data_arg.add_argument('--glass_partial_path',type = str, default = './data/glass/glass_partial.csv', help = 'path of glass partial data')
data_arg.add_argument('--segmentation_path',type = str, default = './data/segmentation/segmentation.data', help = 'path of glass data')
data_arg.add_argument('--segmentation_test_path',type = str, default = './data/segmentation/segmentation.test', help = 'path of glass test data')
data_arg.add_argument('--segmentation_partial_path',type = str, default = './data/segmentation/segmentation_partial.csv', help = 'path of glass partial data')



if __name__ == '__main__':
    config, _ = get_config()
    Generate_Data(config)  ## 生成数据
