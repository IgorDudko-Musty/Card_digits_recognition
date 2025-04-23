import os
import shutil
import numpy as np 
import random
import cv2
import albumentations as A
import yaml
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(r"../logs/datacreator.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    logger.info("DATA GENERATION START\n")
    parser = argparse.ArgumentParser(description="Synthetic data creation")
    parser.add_argument('--par_dir', 
                        default=r'../parameters/parameters_datacreator.yml',
                        type=str, 
                        help='path to the parameter yaml file')
     
    args = parser.parse_args()
     
    with open(args.par_dir, 'r') as f:
        par_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    logger.info(f"First didgits exsample path: {par_dict['path_to_examples1']}")
    logger.info(f"Second didgits exsample path: {par_dict['path_to_examples2']}")
    logger.info(f"Train data path: {par_dict['tr_data_path']}")
    logger.info(f"Validation data path: {par_dict['val_data_path']}")
    logger.info(f"Test data path: {par_dict['test_data_path']}")
    logger.info(f"Train data size for each digits example: {par_dict['train_size']}")
    logger.info(f"Validation data size for each digits example: {par_dict['val_size']}")
    logger.info(f"Test data size for each digits example: {par_dict['test_size']}")
    logger.info(f"Kind of augmentation: {par_dict['simple_aug']}")
    logger.info(f"Image of digit size: {par_dict['dim']}")    
    
    DataCreator(path_to_examples1=par_dict['path_to_examples1'],
                path_to_examples2=par_dict['path_to_examples2'],
                tr_data_path=par_dict['tr_data_path'],
                val_data_path=par_dict['val_data_path'],
                test_data_path=par_dict['test_data_path'],
                train_size=par_dict['train_size'],
                val_size=par_dict['val_size'],
                test_size=par_dict['test_size'],
                simple_aug=par_dict['simple_aug'],
                dim=par_dict['dim'])
    
    logger.info("DATA GENERATION END\n")

class DataCreator():
    def __init__(self,
                 path_to_examples1=r'../digit_examples/creditcard_digits1.jpg',
                 path_to_examples2=r'../digit_examples/creditcard_digits2.jpg',
                 tr_data_path=r"../data/train",
                 val_data_path=r"../data/validation",
                 test_data_path=r"../data/test",
                 train_size=4000,
                 val_size=1000,
                 test_size=100,
                 simple_aug=False,
                 dim=32):
        self.digits_ex1 = 255 - cv2.imread(path_to_examples1, cv2.IMREAD_GRAYSCALE)
        self.digits_ex2 = cv2.imread(path_to_examples2, cv2.IMREAD_GRAYSCALE)
        self.tr_data_path = tr_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.digits_ex2[self.digits_ex2 <= 173] = 0
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.simple_aug = simple_aug
        self.dim = dim
        self.data_aug = A.Compose([
            # A.HorizontalFlip(p=0.5),
            A.OneOf([
                    A.Affine(
                        scale=(0.8, 1.2),     
                        rotate=(-10, 10),     
                        translate_percent=(-0.1, 0.1), 
                        shear=(-10, 10),         
                        p=0.8 
                            ),
                    A.Perspective(scale=(0.05, 0.1), p=0.8) 
                    ], p=0.7),
            A.OneOf([
                    A.GaussianBlur(blur_limit=(15, 15), p=0.5),
                    ], p=0.3),
                    ])
        self.makedir()
        self.create_train()
        self.create_validate()
        # self.create_test()
                   
    
    def create_train(self):

        region = [(2, 19), (50, 72)]

        top_left_y = region[0][1]
        bottom_right_y = region[1][1]
        top_left_x = region[0][0]
        bottom_right_x = region[1][0]
        
        number_arr_ex1 = np.zeros((10, self.train_size, self.dim, self.dim),
                                  dtype=np.float32)
        for i in range(0,10):   
            # We jump the next digit each time we loop
            if i > 0:
                top_left_x = top_left_x + 59
                bottom_right_x = bottom_right_x + 59

            roi = self.digits_ex1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            print("Augmenting Digit - ", str(i))
            # We create 200 versions of each image for our dataset
            for j in range(0, self.train_size):
                if self.simple_aug:
                    roi2 = self.digit_augmentation(roi)
                    number_arr_ex1[i, j] = roi2 / 255.
                else:
                    roi2 = self.data_aug(image=roi)
                    roi2 = cv2.resize(roi2['image'], 
                                      (self.dim, self.dim), 
                                      interpolation = cv2.INTER_AREA)
                    number_arr_ex1[i, j] = roi2 / 255.
        # Creating 2000 Images for each digit in creditcard_digits2 - TRAINING DATA
       
        region = [(0, 0), (35, 48)]

        top_left_y = region[0][1]
        bottom_right_y = region[1][1]
        top_left_x = region[0][0]
        bottom_right_x = region[1][0]
        
        number_arr_ex2 = np.zeros((10, self.train_size, self.dim, self.dim),
                                  dtype=np.float32)
        for i in range(0,10):   
            if i > 0:
                # We jump the next digit each time we loop
                top_left_x = top_left_x + 35
                bottom_right_x = bottom_right_x + 35

            roi = self.digits_ex2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            print("Augmenting Digit - ", str(i))
            # We create 200 versions of each image for our dataset
            for j in range(0, self.train_size):
                if self.simple_aug:
                    roi2 = self.digit_augmentation(roi)
                    number_arr_ex2[i, j] = roi2 / 255.
                else:
                    roi2 = self.data_aug(image=roi)
                    roi2 = cv2.resize(roi2['image'], 
                                      (self.dim, self.dim), 
                                      interpolation = cv2.INTER_AREA)
                    number_arr_ex2[i, j] = roi2 / 255.
                
        number_arr_full = np.concatenate((number_arr_ex1, number_arr_ex2), axis=1)
        logger.info(f"({self.train_size} samples of first example and {self.train_size} samples of second example) * 10 classes are saved in {self.tr_data_path}")
        for i in range(10):
            with open(os.path.join(self.tr_data_path, str(i), str(i) + '.npy'), 'wb') as f:
               np.save(f, number_arr_full[i])
               # np.save(f, number_arr_ex2[i])
               
    
    def create_validate(self):
        
        region = [(2, 19), (50, 72)]

        top_left_y = region[0][1]
        bottom_right_y = region[1][1]
        top_left_x = region[0][0]
        bottom_right_x = region[1][0]
        
        number_arr_ex1 = np.zeros((10, self.val_size, self.dim, self.dim), 
                                  dtype=np.float32)
        for i in range(0,10):   
            # We jump the next digit each time we loop
            if i > 0:
                top_left_x = top_left_x + 59
                bottom_right_x = bottom_right_x + 59

            roi = self.digits_ex1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            print("Augmenting Digit - ", str(i))
            # We create 200 versions of each image for our dataset
            for j in range(0, self.val_size):
                if self.simple_aug:
                    roi2 = self.digit_augmentation(roi)
                    number_arr_ex1[i, j] = roi2 / 255.
                else:
                    roi2 = self.data_aug(image=roi)
                    roi2 = cv2.resize(roi2['image'], 
                                      (self.dim, self.dim), 
                                      interpolation = cv2.INTER_AREA)
                    number_arr_ex1[i, j] = roi2 / 255.
                
        # Creating 2000 Images for each digit in creditcard_digits2 - TRAINING DATA

        region = [(0, 0), (35, 48)]

        top_left_y = region[0][1]
        bottom_right_y = region[1][1]
        top_left_x = region[0][0]
        bottom_right_x = region[1][0]
        
        number_arr_ex2 = np.zeros((10, self.val_size, self.dim, self.dim),
                                  dtype=np.float32)
        for i in range(0,10):   
            if i > 0:
                # We jump the next digit each time we loop
                top_left_x = top_left_x + 35
                bottom_right_x = bottom_right_x + 35

            roi = self.digits_ex2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            print("Augmenting Digit - ", str(i))
            # We create 200 versions of each image for our dataset
            for j in range(0, self.val_size):
                if self.simple_aug:
                    roi2 = self.digit_augmentation(roi)
                    number_arr_ex2[i, j] = roi2 / 255.
                else:
                    roi2 = self.data_aug(image=roi)
                    roi2 = cv2.resize(roi2['image'], 
                                      (self.dim, self.dim), 
                                      interpolation = cv2.INTER_AREA)
                    number_arr_ex2[i, j] = roi2 / 255.
               
        number_arr_full = np.concatenate((number_arr_ex1, number_arr_ex2), axis=1)
        logger.info(f"({self.val_size} samples of first example and {self.val_size} samples of second example) * 10 classes are saved in {self.val_data_path}")
        for i in range(10):
            with open(os.path.join(self.val_data_path, str(i), str(i) + '.npy'), 'wb') as f:
               np.save(f, number_arr_full[i])
               # np.save(f, number_arr_ex2[i])
               
                
    def create_test(self):
       
        region = [(2, 19), (50, 72)]

        top_left_y = region[0][1]
        bottom_right_y = region[1][1]
        top_left_x = region[0][0]
        bottom_right_x = region[1][0]
        
        number_arr_ex1 = np.zeros((10, self.test_size, self.dim, self.dim),
                                  dtype=np.float32)
        for i in range(0,10):   
            # We jump the next digit each time we loop
            if i > 0:
                top_left_x = top_left_x + 59
                bottom_right_x = bottom_right_x + 59

            roi = self.digits_ex1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            print("Augmenting Digit - ", str(i))
            # We create 200 versions of each image for our dataset
            for j in range(0, self.test_size):
                if self.simple_aug:
                    roi2 = self.digit_augmentation(roi)
                    number_arr_ex1[i, j] = roi2 / 255.
                else:
                    roi2 = self.data_aug(image=roi)
                    roi2 = cv2.resize(roi2['image'], 
                                      (self.dim, self.dim), 
                                      interpolation = cv2.INTER_AREA)
                    number_arr_ex1[i, j] = roi2 / 255.
               
        # Creating 2000 Images for each digit in creditcard_digits2 - TRAINING DATA

        region = [(0, 0), (35, 48)]

        top_left_y = region[0][1]
        bottom_right_y = region[1][1]
        top_left_x = region[0][0]
        bottom_right_x = region[1][0]
        
        number_arr_ex2 = np.zeros((10, self.test_size, self.dim, self.dim),
                                  dtype=np.float32)
        for i in range(0,10):   
            if i > 0:
                # We jump the next digit each time we loop
                top_left_x = top_left_x + 35
                bottom_right_x = bottom_right_x + 35

            roi = self.digits_ex2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            print("Augmenting Digit - ", str(i))
            # We create 200 versions of each image for our dataset
            for j in range(0, self.test_size):
                if self.simple_aug:
                    roi2 = self.digit_augmentation(roi)
                    number_arr_ex2[i, j] = roi2 / 255.
                else:
                    roi2 = self.data_aug(image=roi)
                    roi2 = cv2.resize(roi2['image'], 
                                      (self.dim, self.dim), 
                                      interpolation = cv2.INTER_AREA)
                    number_arr_ex2[i, j] = roi2 / 255.
               
        number_arr_full = np.concatenate((number_arr_ex1, number_arr_ex2), axis=1)
        logger.info(f"({self.test_size} samples of first example and {self.test_size} samples of second example) * 10 classes are saved in {self.test_data_path}")
        for i in range(10):
            with open(os.path.join(self.test_data_path, str(i), str(i) + '.npy'), 'wb') as f:
               np.save(f, number_arr_full[i])
        
        
               
    def digit_augmentation(self, frame):
        """Randomly alters the image using noise, pixelation and streching image functions"""
        frame = cv2.resize(frame, 
                           None, 
                           fx=2, 
                           fy=2, 
                           interpolation = cv2.INTER_CUBIC)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        random_num = np.random.randint(0,9)

        if (random_num % 2 == 0):
            frame = self.add_noise(frame)
        if(random_num % 3 == 0):
            frame = self.pixelate(frame)
        if(random_num % 2 == 0):
            frame = self.stretch(frame)
        frame = cv2.resize(frame, 
                           (self.dim, self.dim), 
                           interpolation = cv2.INTER_AREA)
        return frame 

    def add_noise(self, image):
        """Addings noise to image"""
        prob = random.uniform(0.01, 0.05)
        rnd = np.random.rand(image.shape[0], image.shape[1])
        noisy = image.copy()
        noisy[rnd < prob] = 0
        noisy[rnd > 1 - prob] = 1
        return noisy

    def pixelate(self, image):
        "Pixelates an image by reducing the resolution then upscaling it"
        dim = np.random.randint(8,12)
        image = cv2.resize(image, (dim, dim), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (16, 16), interpolation = cv2.INTER_AREA)
        return image

    def stretch(self, image):
        "Randomly applies different degrees of stretch to image"
        ran = np.random.randint(0,3)*2
        if np.random.randint(0,2) == 0:
            frame = cv2.resize(image, (32, ran + 32), interpolation = cv2.INTER_AREA)
            return frame[int(ran/2):int(ran + 32)-int(ran/2), 0:32]
        else:
            frame = cv2.resize(image, (ran + 32, 32), interpolation = cv2.INTER_AREA)
            return frame[0:32, int(ran/2):int(ran + 32)-int(ran/2)]
    
    def makedir(self):
        """Creates a new directory if it does not exist"""
        try:
            if os.listdir(self.tr_data_path):
                shutil.rmtree(self.tr_data_path)
            for i in range(0,10):
                directory_name = os.path.join(self.tr_data_path, str(i))
                print(directory_name)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
        except:
            for i in range(0,10):
                directory_name = os.path.join(self.tr_data_path, str(i))
                print(directory_name)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
        try:
            if os.listdir(self.val_data_path):
                shutil.rmtree(self.val_data_path)
            for i in range(0,10):
                directory_name = os.path.join(self.val_data_path, str(i))
                print(directory_name)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
        except:
            for i in range(0,10):
                directory_name = os.path.join(self.val_data_path, str(i))
                print(directory_name)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
        try:
            if os.listdir(self.test_data_path):
                shutil.rmtree(self.test_data_path)
            for i in range(0,10):
                directory_name = os.path.join(self.test_data_path, str(i))
                print(directory_name)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
        except:
            for i in range(0,10):
                directory_name = os.path.join(self.test_data_path, str(i))
                print(directory_name)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)

        
if __name__ == "__main__":
    
    main()        







