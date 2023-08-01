import numpy as np

class NetworkTrainer:
    def __init__(self, neural_network):
        self.neural_network = neural_network

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
                self.sgd(data_batch, labels_batch, learning_rate)

                model_labels = self.neural_network.forward(data_batch)
                self.neural_network.backward(data_batch, labels_batch, model_labels, learning_rate)

                loss = self.neural_network.calculate_loss(data_train, labels_train)

                model_label = self.neural_network.predict(data_test)
                accuracy = np.mean(model_label == np.argmax(labels_test, axis=1))

                epoch_print = f"Epoch {epoch + 1}/{epochs}"
                step_print = f"Step: {int(batch / batch_size)}"
                loss_print = f"Loss: {loss:.4f}"
                test_print = f"Accuracy: {accuracy:.4f}"

                print(epoch_print, step_print, loss_print, test_print)