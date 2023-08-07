import numpy as np
import pandas as pd

class NetworkTrainer:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.results = pd.DataFrame(columns=("activation",
                                             "hl arch",
                                             "epoch",
                                             "step",
                                             "loss",
                                             "accuracy"))
        
        self.activation = None
        self.hl_arch = None
        self.epoch = None
        self.step = None
        self.loss = None
        self.accuracy = None

    def sgd(self, data, truth_labels, learning_rate):
        model_labels = self.neural_network.forward(data)
        self.neural_network.backward(data, truth_labels, model_labels, learning_rate)

    def train_network(self,
                      data_train, labels_train,
                      data_test, labels_test,
                      learning_rate=0.1,
                      epochs=1,
                      batch_size=32):

        for epoch in range(epochs):
            array = data_train.shape[0]
            batches = np.arange(array)
            np.random.shuffle(batches)

            for batch in range(0, array, batch_size):
                eob = batch + batch_size
                mini_batch = batches[batch:eob]
                data_batch = data_train[mini_batch]
                labels_batch = labels_train[mini_batch]

                model_labels = self.neural_network.forward(data_batch)
                self.sgd(data_batch, labels_batch, learning_rate)
                self.neural_network.backward(data_batch, labels_batch, model_labels, learning_rate)

                loss = self.neural_network.calculate_loss(data_train, labels_train)

                model_label = self.neural_network.predict(data_test)
                accuracy = np.mean(model_label == np.argmax(labels_test, axis=1))
                
                self.activations = self.neural_network.get_activation()
                self.hl_arch = self.neural_network.get_hidden_layers()
                self.epoch = int(epoch + 1/epochs)
                self.step = int(batch / batch_size)
                self.loss = round(loss, 4)
                self.accuracy = round(accuracy, 4)
                
                self.print_save()
                                    
                
    def print_save(self):
        
        activation_print = f"Activation: {self.activations}"
        epoch_print = f"Epoch: {self.epoch}"
        step_print = f"Step: {self.step}"
        loss_print = f"Loss: {self.loss}"
        accuracy_print = f"Accuracy: {self.accuracy}"
        
        results = {'activation' : self.activations,
                   'hl arch' : self.hl_arch,
                   'epoch' : self.epoch,
                   'step' : self.step,
                   'loss' : self.loss,
                   'accuracy' : self.accuracy}
        
        self.results = self.results.append(results, ignore_index=True)

        print(activation_print, epoch_print, step_print, loss_print, accuracy_print)
        
    def get_results(self):
        return self.results
        