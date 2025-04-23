import yaml
import argparse
import pandas as pd
import os
from nnet import Model_Predict
from card_image_proc import Image_Proc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(r"../predict.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('--par_dir',
                        default=r'../parameters/parameters_infer.yml',
                        type=str,
                        help='path to the parameter yaml file')
     
    args = parser.parse_args()
     
    with open(args.par_dir, 'r') as f:
        par_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
         
    im_proc = Image_Proc(images_path=par_dict['path_to_data'])
    digits = im_proc.digit_extract()
    model = Model_Predict(path_to_model=par_dict['path_to_model'],
                          nn_type=par_dict['nn_type'],
                          dropout=par_dict['dropout'],
                          device=par_dict['device'],
                          need_transform=par_dict['need_transform'])
    
    with open(r'../data_for_test/answers.csv', 'r') as f:
        answers = pd.read_csv(f, sep=';')
            
    true_answer = answers.numbers[answers.image \
                                  == os.path.split(par_dict['path_to_data'])[-1]]
    digits_list = []
    for digit in digits:
        if isinstance(digit, str):
            continue
        digits_list.append(model.predict(digit))
    logger.info(f"Image answer: {os.path.split(par_dict['path_to_data'])[-1]}")
    logger.info(f"True answer: {true_answer.iloc[0]}")
    logger.info(f"Predicted answer: {digits_list}\n\n")
        

if __name__ == "__main__":
    
    main()