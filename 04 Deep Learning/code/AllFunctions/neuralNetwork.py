import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self):

        # list which will contain "losses by iteration" after training
        self.loss = []
        # the layer providing the data (holds input data layerfor NN)
        self.data_layer = None
        # The layer calculating the loss and the prediction(holds loss layer for calculating the network's loss)
        self.loss_layer = None
        # The defination of the particular neural network (holds the layers of NN)
        self.layers = []

    # The forward pass of the network, returning activation of the last layer
    def forward(self, activation_tensor):

        # Pass the input up the network(through each layer)
        for layer in self.layers: #iterate through each layer in self.layers list.
            activation_tensor = layer.forward(activation_tensor) # calls the forward method of each layer, passing "activation_tensor" and updating it with the result

        # return activation of last layer
        return self.loss_layer.forward(activation_tensor)

    # The raw backward pass during training
    def backward(self, label_tensor):

        # Fetch the label from data layer and pass it through the loss
        error_tensor = self.loss_layer.backward(label_tensor)

        # Pass back the error recursively
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    # High level method to train the network
    def train(self, iterations):

        # Iterate for a fixed number of steps
        for i in range(iterations):
            
            # get the training data
            input_tensor = self.data_layer.forward()
            label_tensor = self.data_layer.backward()

            # Pass the input up the network
            self.forward(input_tensor)

            # Calculate the loss of the network using the loss layer and save it
            estimated_loss = self.loss_layer.loss(label_tensor)
            self.loss.append(estimated_loss)

            # Down the network including update of weights
            self.backward(label_tensor)
    
    # High level method to test a new input
    def test(self, input_tensor):

        return self.forward(input_tensor)
    
    # Plot the loss curve over iterations
    def show(self):
        plt.plot(self.loss)
        plt.show()
    