#Imports - using from * import notation to reduce load/memory
from torch.nn import Linear, Dropout, functional, CrossEntropyLoss, Module
from torch.optim import SGD


class CreateNetwork(Module):
    
    #Defining Neural Network
    def __init__(self):
        super(CreateNetwork, self).__init__()
        
        #Network layers
        self.fc1 = Linear(28 * 28, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 10)
        
        #Overfitting prevention
        self.dropout = Dropout(0.2)
        
        #Literal model collection
        self.model_name = 'data\\model\\model.pt'

    #Forward pass work
    def forward(self, x):
        
        #Preparing image for forward oass
        x = x.view(-1, 28 * 28)
        
        #Hidden layer 1
        x = functional.relu(self.fc1(x))
        
        #Overfitting prevention 1
        x = self.dropout(x)
        
        #Hidden layer 2
        x = functional.relu(self.fc2(x))
        
        #Overfitting prevention 2
        x = self.dropout(x)
        
        #Forward pass
        x = self.fc3(x)
        return x

    #Loss function with optimiser
    @staticmethod
    def loss(model, option=None):
        
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        return criterion if option == "c" else optimizer

    #Returning model with neural network attached
    def get_model_name(self):
        return self.model_name
