#Imports - using from * import notation to reduce load/memory
import torch
import numpy as np
from preparation import Preparation
from create_network import CreateNetwork


class TestNetwork:
    
    #Collecting training dataset, model, and neaural network
    def __init__(self, model):
        self.model = model
        prep = Preparation()
        self.network = CreateNetwork()
        self.model_name = self.network.get_model_name()
        self.train_loader = prep.loading(get_value="train")

    #Collecting model with highest validity
    def test(self):
        self.model.load_state_dict(torch.load(self.model_name))
        
        #Collecting loss/accuracy
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        self.model.eval()

        #Forward pass of imputs to model
        for data, target in self.train_loader:
            output = self.model(data)
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            #Loss/accuracy calculation
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    #Returning test model
    def get_tested_model(self):
        TestNetwork.test(self)
        return self.model
