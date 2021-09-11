import os
import sys
os.chdir(r'F:\code\python\data_mining\possc-pytorch')
sys.path.append('F:\code\python\data_mining\possc2.0')

import argparse
from dataset.generation import generation
from dataset.preprocess import preprocess

from train.train_nn import train
from pred.pred_nn import predict, test
from pred.pred_lr import linearRegression
from utils.clean import *

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-t', '--train_model', nargs=4,
                        help="Train network model from steelmaking dataset")
    parse.add_argument('-p', '--predict', nargs=4,
                        help="Predect from the trained model that named 0x*")
    parse.add_argument('-l', '--linear_regression', nargs=5,
                        help='Linear regression')
    parse.add_argument('-c', '--clean_steelTypeCache', nargs=2,
                        help='Delete saved model or dataset')
    parse.add_argument('-n', '--clean_steelTypeCodeCache', nargs=3, 
                        help="Cleans the cache for a specific code under a given steel class")
    parse.add_argument('-g', '--generate_data', action='store_true', 
                        help="Generate native data from source_path")
    parse.add_argument('-s', '--process_data', nargs=4,
                        help='Data preprocess from generated data')

    parse.add_argument('-e', '--test_model', nargs=3, 
                        help='Test the model from the dataset')
    parse.add_argument('-r', '--system_reset', action='store_true', 
                        help="reset system, cleaning up cached data")

    args = parse.parse_args()
    if args.train_model is not None:
        steel_type = args.train_model[0]
        stove_num = args.train_model[1]
        input_columns = args.train_model[2].split(',')
        output_columns = args.train_model[3].split(',')
        train(steel_type, stove_num, input_columns, output_columns)
    elif args.generate_data:
        generation()
    elif args.process_data is not None:
        steelType = args.train_model[0]
        stoveNum = args.train_model[1]
        input_factors = args.train_model[2]
        output_factors = args.train_model[3]
        preprocess(steelType, stoveNum, input_factors, output_factors)
    elif args.predict is not None:
        steelType = args.predict[0]
        stoveNum = args.predict[1]
        input_factorNum = args.predict[2].split(',')
        code_16 = args.predict[3]
        predict(steelType, stoveNum, input_factorNum, code_16)
    elif args.test_model is not None:
        steelType = args.test_model[0]
        stoveNum = args.test_model[1]
        code_16 = args.test_model[2]
        test(steelType, stoveNum, code_16)
    elif args.linear_regression is not None:
        steelType = args.linear_regression[0]
        stoveNum = args.linear_regression[1]
        input_factors = args.linear_regression[2].split(',')
        output_factors = []
        output_factors.append(args.linear_regression[3])
        input_factorNum = args.linear_regression[4].split(',')
        linearRegression(steelType, stoveNum, input_factors, output_factors, input_factorNum)
    elif args.system_reset:
        sysReset()
    elif args.clean_steelTypeCache is not None:
        steelType = args.clean_steelTypeCache[0]
        stoveNum = args.clean_steelTypeCache[1]
        rm_steelTypeAllFile(steelType)
    elif args.clean_steelTypeCodeCache is not None:
        steelType = args.clean_steelTypeCodeCache[0]
        stoveNum = args.clean_steelTypeCodeCache[1]
        code_16 = args.clean_steelTypeCodeCache[2]
        rm_dataFile(steelType, stoveNum, code_16)

        





        
