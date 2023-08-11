#Imports - using from * import notation to reduce load/memory
import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class Preparation:

    #Paramitisation of model
    def __init__(self):
        self.num_workers = 0
        self.batch_size = 20
        self.valid_size = 0.2
        self.transformer = ToTensor()

    #Collecting training data subset
    def train_dataset(self):
        return datasets.FashionMNIST(root='data',
                              train=True,
                              download=True,
                              transform=self.transformer)
    
    #Collecting testing data subset
    def test_dataset(self):
        return datasets.FashionMNIST(root='data',
                                   train=False,
                                   download=True,
                                   transform=self.transformer)
    
    #Training model
    def training(self):
        
        #Collecting training paramater
        num_train = len(Preparation.train_dataset(self))
        indices = list(range(num_train))
        
        #Randomisation of training indicies
        np.random.shuffle(indices)
        
        #Splitting the training model using the defined parameter
        split = int(np.floor(self.valid_size * num_train))
        
        return indices[split:], indices[:split]

    #Sampling the trained model
    def sampling(self):
        
        #Collecting training indicies
        train_index, valid_index = self.training()
        
        #Randomly collecting indicies from train/valid models
        train_index = SubsetRandomSampler(train_index)
        valid_index = SubsetRandomSampler(valid_index)
        
        return train_index, valid_index

    #Helper function to return train/valid/test models to network
    def loading(self, get_value):

        train_sampler, valid_sampler = Preparation.sampling(self)
        train_dataset = Preparation.train_dataset(self)
        test_dataset = Preparation.test_dataset(self)

        def get_train():
            return DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              sampler=train_sampler,
                              num_workers=self.num_workers)

        def get_valid():
            return DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              sampler=valid_sampler,
                              num_workers=self.num_workers)

        def get_test():
            return DataLoader(test_dataset,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)

        if get_value == "train":
            return get_train()
        
        elif get_value == "valid":
            return get_valid()
        
        elif get_value == "test":
            return get_test()
