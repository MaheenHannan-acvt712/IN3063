import numpy as np
from torchvision import datasets

class DatasetGenerator:
    def __init__(self):
        self.num_workers = 0
        self.batch_size = 20
        self.valid_size = 0.2
        
    def dataset(self, train=True):
        return datasets.MNIST(root='data',
                              train=train,
                              download=True)
        
    def to_numpy(self, dataset):
        data = dataset.data.numpy()
        labels = dataset.targets.numpy()
        data = np.squeeze(np.array(data))
        labels = np.squeeze(np.array(labels))
        data = data / 255.0
        data_flat = data.reshape(data.shape[0], -1)
        labels = np.eye(10)[labels]
        return data_flat, labels

        
    def get_train_data(self):
        return self.to_numpy(self.dataset())

    def get_test_data(self):
        return self.to_numpy(self.dataset(False))
    
    def get_layers(self):
        data = self.dataset().data[0].shape[0]
        labels = len(np.unique(self.dataset().targets))
        return (data*data), labels
