import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import logging


class Data_Formation(Dataset):
    """
    Класс, осуществляющий загрузку данных для обучения и валидации
    """
    def __init__(self, 
                 path_to_data=r'../data', 
                 data='train',
                 need_transform=False):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(r"../logs/learning_log.log")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.info("DATA LOADING START\n")
        self.need_transform = need_transform
        
        if self.need_transform:
            logger.info("v2 data transform - to torch tensor, to float32, normaliztion (0.5, 0.5)")
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5,), std=(0.5,))                         
                ])
        logger.info("Data is not transform by v2")
        self.data_list = []
        if data == 'train':
            self.path = os.path.join(path_to_data, data)
        elif data == 'validation':
            self.path = os.path.join(path_to_data, data)
        else:
            self.path = os.path.join(path_to_data, data)
        classes_list = os.listdir(self.path)
        self.target_mat = torch.eye(len(classes_list), dtype=torch.float32)

        self.data = None
        # т.к. данных немного - сохраняем и загружаем их одним файлом на один класс цифр
        for cl in classes_list:
            with open(os.path.join(self.path, cl, cl + '.npy'), 'rb') as f:
                if self.data is None:
                    self.data = np.load(f)
                    self.targets = np.full(shape=self.data.shape[0],
                                           fill_value=int(cl))        
                else:
                    self.data = np.concatenate((self.data, np.load(f)),
                                                   axis=0)
                    self.targets = np.concatenate((self.targets,
                                                   np.full(shape=(self.data.shape[0] - self.targets.shape[0]),
                                                   fill_value=int(cl))))
        logger.info(f"The {data} data is loaded. Image amount is {self.data.shape[0]}")
        logger.info("DATA LOADING END\n")

    def __getitem__(self, index):
        if self.need_transform:
            return self.transform(self.data[index]), \
                self.target_mat[self.targets[index]]
        return torch.from_numpy(self.data[index]).unsqueeze(0), \
            self.target_mat[self.targets[index]]

    def __len__(self):
        return self.data.shape[0]
