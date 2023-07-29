from dataset_generator import DatasetGenerator
from neural_network import NeuralNetwork
from network_trainer import NetworkTrainer

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset_generator = DatasetGenerator()
data_train, labels_train = dataset_generator.get_train_data()
data_test, labels_test = dataset_generator.get_test_data()

layers = [128, 64]
neural_network = NeuralNetwork(layers, activation='sigmoid')

network_trainer = NetworkTrainer(neural_network)
network_trainer.train_network(data_train, labels_train, data_test, labels_test)
