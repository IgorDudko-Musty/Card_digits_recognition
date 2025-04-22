# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:37:47 2025

@author: iGOR
"""

import yaml
import argparse
from nnet import Model_Learning
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(r"./logs/learning_log.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    logger.info("THE LEARNING PROCEDURE START\n")
    parser = argparse.ArgumentParser(description="Learning start")
    parser.add_argument('--par_dir', 
                        default=r'./parameters/parameters_learning.yml',
                        type=str, 
                        help='path to the parameter yaml file')
     
    args = parser.parse_args()
     
    with open(args.par_dir, 'r') as f:
        par_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    logger.info(f"The train and validation data path: {par_dict['data_path']}")
    logger.info(f"The model save path: {par_dict['model_path']}")
    logger.info(f"The model name: {par_dict['model_name']}")
    logger.info(f"The batch size: {par_dict['batch_size']}")
    logger.info(f"The dropout percent: {par_dict['dropout']}")
    logger.info(f"NN architecture type: {par_dict['nn_type']}")
    logger.info(f"Working device type: {par_dict['device']}")
    logger.info(f"Epoch amount: {par_dict['EPOCHS']}")
    logger.info(f"The v2 transformation used: {par_dict['need_transform']}\n") 
    
    model = Model_Learning(data_path=par_dict['data_path'],
                           model_path=par_dict['model_path'],
                           model_name=par_dict['model_name'],
                           batch_size=par_dict['batch_size'],
                           dropout=par_dict['dropout'],
                           nn_type=par_dict['nn_type'],
                           device=par_dict['device'],
                           EPOCHS=par_dict['EPOCHS'],
                           need_transform=par_dict['need_transform'])
    logger.info("Fit start\n")
    model.start_fit()
    logger.info("THE LEARNING PROCEDURE END\n")


if __name__ == "__main__":
    
    main()