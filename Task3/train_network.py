#Imports - using from * import notation to reduce load/memory
from create_network import CreateNetwork
from preparation import Preparation
import torch


class TrainNetwork:
    
    #Collecting model, train/valid datasets
    def __init__(self, model):
        self.model = model
        prep = Preparation()
        self.network = CreateNetwork()
        self.train_loader = prep.loading(get_value="train")
        self.valid_loader = prep.loading(get_value="valid")
        self.model_name = self.network.get_model_name()
        
        #Defining training iterations
        self.epochs = 10

    #Model training function
    def train(self):
        
        #Collecting optimiser and criterion functions
        optimizer = self.network.loss(self.model)
        criterion = self.network.loss(self.model, option="c")
        
        #Setting initial loss to infinity to keep invariant
        valid_loss_min = float('infinity')

        for epoch in range(self.epochs):
            print("epoch:", epoch)
            
            #Loss track
            valid_loss = 0
            
            #Preparing model for training
            self.model.train()
            for data, label in self.train_loader:
                
                #Zero gradient optimiser with forward pass and loss calculation
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, label)
                
                #Backward pass with optimisation
                loss.backward()
                optimizer.step()

            #Preparing model for validation
            self.model.eval()
            for data, label in self.valid_loader:
                
                #Forward pass with loss calculation update
                output = self.model(data)
                loss = criterion(output, label)
                valid_loss = loss.item() * data.size(0)

            #Update model only if validation has increased
            valid_loss = valid_loss / len(self.valid_loader.sampler)
            if valid_loss <= valid_loss_min:
                torch.save(self.model.state_dict(), self.model_name)
                valid_loss_min = valid_loss

    #Returning trained model
    def get_trained_model(self):
        TrainNetwork.train(self)
        return self.model
