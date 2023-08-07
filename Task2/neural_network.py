import numpy as np
from dataset_generator import DatasetGenerator

def get_relu(func, activation=False, gradient=False):
    if activation and not gradient:
        relu = np.maximum(0, func)
        return relu
    elif not activation and gradient:
        gradient = np.where(func > 0, 1, 0)
        return gradient
    else:
        return None


def get_sigmoid(func, activation=False, gradient=False):
    sigmoid = 1 / (1 + np.exp(-func))
    if activation and not gradient:
        return sigmoid
    elif not activation and gradient:
        gradient = sigmoid * (1 - sigmoid)
        return gradient
    else:
        return None


def get_softmax(func=None, model_labels=None, truth_labels=None, activation=False, gradient=False):
    batch_size = model_labels.shape[0]
    gradient = (model_labels - truth_labels) / batch_size
    return gradient


class NeuralNetwork:
    def __init__(self, hidden_layers, activation):
        input_layers, output_layers = DatasetGenerator().get_layers()
        self.layers = [input_layers, hidden_layers[0],
                       hidden_layers[1], output_layers]
        
        self.hidden_layers = hidden_layers
        
        self.num_layers = len(self.layers)
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(self.num_layers - 1)]
        self.activation = activation
        self.activations = []
        self.vect_transfer_list = []

        self.weights = []
        for layer in range(self.num_layers - 1):
            random_matrix = np.random.randn(self.layers[layer], self.layers[layer + 1])
            scale = np.sqrt(2 / self.layers[layer])
            self.weights.append(random_matrix * scale)
    
    def get_activation(self):
        return self.activation
    
    def get_hidden_layers(self):
        return self.hidden_layers

    def forward(self, data):
        self.activations = []
        self.vect_transfer_list = []

        outputs = data
        self.activations.append(outputs)

        for layer in range(self.num_layers - 1):
            i = np.dot(outputs, self.weights[layer])
            vect_transfer = i + self.biases[layer]
            self.vect_transfer_list.append(vect_transfer)

            if self.activation == 'sigmoid':
                outputs = get_sigmoid(vect_transfer, activation=True)
            elif self.activation == 'relu':
                outputs = get_relu(vect_transfer, activation=True)

            self.activations.append(outputs)

        return outputs

    def backward(self, data, truth_labels, model_labels, learning_rate):
        batch_size = data.shape[0]
        gradients_weights = []
        gradients_biases = []

        if self.activation == 'sigmoid':
            activation_gradient = get_sigmoid
        elif self.activation == 'relu':
            activation_gradient = get_relu

        # Backpropagation
        delta = get_softmax(model_labels=model_labels, truth_labels=truth_labels, gradient=True)

        for layer in range(self.num_layers - 1, 0, -1):
            output = self.activations[layer - 1]
            vect_transfer = self.vect_transfer_list[layer - 1]

            gradients_vect_transform = activation_gradient(vect_transfer, gradient=True)
            gradients_vect_transform = gradients_vect_transform * delta

            gradients_weight = np.dot(output.T, gradients_vect_transform)
            gradients_weight = gradients_weight / batch_size

            gradients_bias = np.sum(gradients_vect_transform, axis=0, keepdims=True)
            gradients_bias = gradients_bias / batch_size

            delta = np.dot(gradients_vect_transform, self.weights[layer - 1].T)

            gradients_weights.append(gradients_weight)
            gradients_biases.append(gradients_bias)

        gradients_weights.reverse()
        gradients_biases.reverse()

        # Update weights and biases
        for layer in range(self.num_layers - 1):
            self.weights[layer] -= learning_rate * gradients_weights[layer]
            self.biases[layer] -= learning_rate * gradients_biases[layer]

    def calculate_loss(self, data, truth_labels):
        model_labels = self.forward(data)
        loss = -np.sum(truth_labels * np.log(model_labels + 1e-10)) / data.shape[0]
        return loss

    def predict(self, data):
        model_label = self.forward(data)
        return np.argmax(model_label, axis=1)
