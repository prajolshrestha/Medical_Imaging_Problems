import numpy as np

class FullyConnected:

    def __init__(self, input_size, output_size, batch_size, delta):

        # Initialize the weights randomly
        self.weights = np.random.rand(output_size, input_size + 1)

        # Save the current activation - used during backpropagation
        self.activations = np.zeros((input_size +1, batch_size))

        # Allow individual learning rates per hidden layer
        self.delta = delta

    def forward(self, input_tensor):
        
        # Add a row of ones to the input_tensor for the bias term
        # (such that w^T * x + b becomes w^T * x equivalently)
        input_with_bias = np.vstack((input_tensor, np.ones((1, input_tensor.shape[1])))) # one row added
        self.activations = input_with_bias # save for use during backpropogation

        # Calculate the weighted sum of inputs and apply activation function
        layer_output = np.dot(self.weights, input_with_bias)

        return layer_output

    def backward(self,error_tensor):

        # Update the layer using the learning rate and E * X^T
        # Where E is the error from higher layers and 
        #       X are the stored activations from forward pass

        # 1. Calculate the error for the next layers using E * X^T 
        gradient = np.dot(error_tensor,self.activations.T)

        # 2. Update the weights using the learning rate and the gradient
        self.weights = self.weights - self.delta * gradient

        # Calculate the error tensor for the previous layer
        # Exclude the bias term in the weights
        error_tensor_new = np.dot(self.weights[:,:-1].T, error_tensor)

        return error_tensor_new

    