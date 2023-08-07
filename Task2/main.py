from dataset_generator import DatasetGenerator
from neural_network import NeuralNetwork
from network_trainer import NetworkTrainer
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
frames = []
activations = ['relu', 'sigmoid']
hl_arch_var = [[32, 64], [64, 128], [32, 128], [64, 64], [64, 32], [128, 64]]

if __name__ == "__main__":

    for a in activations:
        for hl in hl_arch_var:
            
            dataset_generator = DatasetGenerator()
            data_train, labels_train = dataset_generator.get_train_data()
            data_test, labels_test = dataset_generator.get_test_data()
            
            
            neural_network = NeuralNetwork(hl, activation=a)
        
            network_trainer = NetworkTrainer(neural_network)
            network_trainer.train_network(data_train, labels_train,
                                          data_test, labels_test,
                                          epochs=10)
            
            frames.append(network_trainer.get_results())
            
    results = pd.concat(frames)
    results.to_csv("results.csv")