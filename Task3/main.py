#Imports - using from * import notation to reduce load/memory
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_network import TrainNetwork
from test_network import TestNetwork
from preparation import Preparation
from create_network import CreateNetwork
import warnings

#"Preventing" version and local systems issues
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Main:
    
    #Collecting test dataset and neural network
    def __init__(self):
        self.prep = Preparation()
        self.model = CreateNetwork()
        self.test_loader = self.prep.loading(get_value="test")

    #Adding test and training models to model
    def train_test(self):
        trained_network = TrainNetwork(self.model)
        self.model = trained_network.get_trained_model()
        tested_network = TestNetwork(self.model)
        self.model = tested_network.get_tested_model()

    #Function to export results from learning model with actual results
    def export_results(self):
        data_iter = iter(self.test_loader)
        images, labels = data_iter.next()
        output = self.model(images)
        _, preds = torch.max(output, 1)
        images = images.numpy()
        
        #Prettifying export
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                         color=("green" if preds[idx] == labels[idx] else "red"))

        plt.savefig("results\\results")

#Executing model training application
if __name__ == '__main__':
    main = Main()
    main.train_test()
    main.export_results()





