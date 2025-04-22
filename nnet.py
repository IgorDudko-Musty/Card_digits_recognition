# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:42:32 2025

@author: iGOR
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from dataloader import Data_Formation
import matplotlib.pyplot as plt
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(r"./logs/learning_log.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)

# [64, 64, 64, "MP", 64, 64, 64, "MP", 64, 64, 64, "MP"]
class Nnet(nn.Module):
    nnets_type = {
        '1': [32,32, "MP", 32,32, "MP", 32,32, "MP"],
        '2': [32, 32, "MP", 128, 128, "MP", 512, 512, "MP"],
        '3': [32, 32, 64, "MP", 128, 128, 256, "MP", 512, 512, 512, "MP"]
        }

    def __init__(self, 
                 nn_type, 
                 dropout=0.25):
        super().__init__()
        self.nnets_type = self.nnets_type[nn_type]
        self.dropout = dropout
        
        self.features = self.features_part(self.nnets_type)
        self.classification = self.classification_part()
        self.flatten = nn.Flatten(start_dim=1)
        
    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        x = self.flatten(x)
        output_signal = self.classification(x)
        return output_signal
    
    def features_part(self, nn_type):
        layers = nn.ModuleList()
        input_channels = 1
        for i in range(len(self.nnets_type)):
            if self.nnets_type[i] == "MP":
                layers.add_module("MaxPooling_{}".format(i),
                                  nn.MaxPool2d(2))
            else:
                layers.add_module("Conv2D_{}".format(i),
                                  nn.Conv2d(input_channels,
                                            self.nnets_type[i],
                                            kernel_size=(3,3),
                                            padding=1))
                layers.add_module("Activation_{}".format(i),
                                  nn.LeakyReLU())
                input_channels = self.nnets_type[i]
        return layers
    
    def classification_part(self):
        return nn.Sequential(
            nn.Linear(32 * 4 * 4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
            )


class Model_Learning():
    
    def __init__(self, 
                 data_path=r"./data",
                 model_path=r"./models",
                 model_name=r'model_{}epoch_{:.3f}loss_{:.3f}acc.pt',
                 batch_size=128, 
                 dropout=0.25,
                 nn_type="1",
                 device='cpu',
                 EPOCHS=10,
                 need_transform=False):
        self.batch_size = batch_size
        self.device = device
        self.EPOCHS = EPOCHS
        # загрузка данных для обучения
        self.path_to_model = model_path
        self.model_name = model_name
        train_data = Data_Formation(data_path,
                                         data='train',
                                         need_transform=need_transform)
        val_data = Data_Formation(data_path,
                                  data='validation',
                                  need_transform=need_transform)
        self.train_data_len = len(train_data)
        self.val_data_len = len(val_data)
        
        self.train_load = DataLoader(train_data,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        self.val_load = DataLoader(val_data,
                                   batch_size=self.batch_size,
                                   shuffle=False)
        # создание модели, определение функции потерь для задач классификации и задание оптимизатора.
        self.model = Nnet(nn_type=nn_type,
                          dropout=dropout).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=1e-5,
            weight_decay=1e-3)
        # списки для сохранения значений функций потерь и метрики
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    
    def start_fit(self):
        best_loss = None
        # цикл тренировки и проверки на валидационнах данных
        for epoch in range(self.EPOCHS):
            logger.info(f"EPOCH {epoch} training")
            self.train()
            logger.info(f"EPOCH {epoch} validation")
            self.validation()
            # проверяем, если на новой эпохе значение функции потерь меньше, то сохраняем модель
            if best_loss is None:
                best_loss = self.val_loss[-1]
                # is_empty = os.listdir(self.path_to_model)
                # if len(is_empty) != 0:
                #     os.remove(os.path.join(self.path_to_model, *is_empty))
                torch.save(
                          self.model.state_dict(), 
                          os.path.join(self.path_to_model, 
                                       self.model_name.format(epoch, 
                                                              best_loss, 
                                                              self.val_acc[-1]))
                          )
                
                
            if best_loss > self.val_loss[-1]:
                best_loss = self.val_loss[-1]
                # is_empty = os.listdir(self.path_to_model)
                # if len(is_empty) != 0:
                #     os.remove(os.path.join(self.path_to_model, *is_empty))
                torch.save(
                          self.model.state_dict(), 
                          os.path.join(self.path_to_model, 
                                       self.model_name.format(epoch, 
                                                              best_loss, 
                                                              self.val_acc[-1]))
                          )
            print("TOTAL: Epoch [{}/{}],  train_loss: {:.4f}, \train_acc: {:.4f}, \
                  val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, self.EPOCHS, self.train_loss[-1], 
                                                            self.train_acc[-1], self.val_loss[-1], self.val_acc[-1]))
    
    def train(self):
        # включаем режим обучения и запускаем стандартный цикл обучения
        self.model.train()
        loop_counter = 0
        loss_in_loop = 0
        true_answer = 0
        # для визуализации бегущей полосы и т.д. используем tqdm
        train_loops = tqdm(self.train_load, leave=True)
        for data, targets in train_loops:
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            pred = self.model(data)
            loss = self.loss_func(pred, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # метрику вычисляем куммулятивно - на каждом батче и суммируем с прошлым значаеним
            loop_counter += 1
            loss_in_loop += loss.item()
            mean_loss_train = loss_in_loop / loop_counter
            
            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            train_loops.set_description('TRAIN: mean_loss: {:.4f}, acc: {:.4f}'.format(mean_loss_train, 
                                        true_answer / (loop_counter * self.batch_size)))
        # заносим значения функции потерь и метрики в список на каждой эпохе
        self.train_loss.append(mean_loss_train)  
        self.train_acc.append(true_answer / self.train_data_len)
        logger.info(f"Training loss {mean_loss_train:.3f}, training accuracy {true_answer / self.train_data_len:.3f}")
    
    def validation(self):
        # включаем режим предсказания и запускаем стандартный цикл прогона по валидационным данным
        self.model.eval()
        with torch.no_grad():
            loop_counter = 0
            loss_in_loop = 0
            true_answer = 0
            # для визуализации бегущей полосы и т.д. используем tqdm
            val_loops = tqdm(self.val_load, leave=True)
            for data, targets in val_loops:
                data = data.to(self.device)
                targets = targets.to(self.device)
                # здесь отличие от цикла обучения только в том, что не делаем обратного распространения ошибки и
                # не корректируем значения параметров
                pred = self.model(data)
                loss = self.loss_func(pred, targets)
                # метрику вычисляем куммулятивно - на каждом батче и суммируем с прошлым значаеним
                loop_counter += 1
                loss_in_loop += loss.item()
                mean_loss_val = loss_in_loop / loop_counter
                
                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                val_loops.set_description('VALIDATION: mean_loss: {:.4f}, acc:{:.4f}'.format(mean_loss_val, 
                                          true_answer / (loop_counter * self.batch_size)))
            # заносим значения функции потерь и метрики в список на каждой эпохе
            self.val_loss.append(mean_loss_val)  
            self.val_acc.append(true_answer / self.val_data_len)
            logger.info(f"Validation loss {mean_loss_val:.3f}, Validation accuracy {true_answer / self.val_data_len:.3f}")
            
   
class Model_Predict():
    
    def __init__(self, 
                 path_to_model=r"./models/model_noaug_notr_3epoch_1.463loss_1.000acc.pt",
                 nn_type='1',
                 dropout=0.25,
                 device='cpu',
                 need_transform=False):
            self.device = device
            self.need_transform = need_transform
            if self.need_transform:
                self.transform = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=(0.5,), std=(0.5,))                         
                    ])
            # загрузка обученной модели
            # temp_file_list = os.listdir(path_to_model)
            # for file in temp_file_list:
            #     if os.path.splitext(file)[-1] == '.pt':
                    # self.model_state_dict = torch.load(os.path.join(path_to_model, 
                    #                                                 file), 
                    #                                    map_location=self.device)
            self.model_state_dict = torch.load(path_to_model, 
                                               map_location=self.device,
                                               weights_only=True)        
            self.model = Nnet(nn_type, dropout).to(self.device)
            self.model.load_state_dict(self.model_state_dict)
            
    def predict(self, image):
        """
        Метод, выполняющий предсказание модели по предоставленным данным. 
        Может осуществлять предсказания как для данных из тестового нобор с 
        указание предсказанной и правильно меток, так и для неизвестных данных
        с указанием метки класса, к которому они, как считает модель, 
        принадлежат.
        
        Параметры:
        ----------
        data: torch.tensor, list
            Данные для классификации. Могут быть представлены как тензором pytorch, так и списком.
        """
        if self.need_transform:
            image = self.transform(image)
        # если данные список, преобразуем из в тензор и придаём необходимую форму
        if isinstance(image, torch.Tensor) == False:
            # image = self.transform(image)
            image = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
        # выполняем предсказание для данных
        self.model.eval()
        with torch.no_grad():
            pred = torch.argmax(self.model(image.unsqueeze(0)))
        return pred.item()
       



