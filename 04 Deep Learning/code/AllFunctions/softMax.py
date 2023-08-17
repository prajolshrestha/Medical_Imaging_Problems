import numpy as np

class SoftMax:

    def __init__(self, categories, batch_size):

        # Store current activations - used during backpropogation
        self.activations = np.zeros((categories,batch_size)) # pre- allocation

    def forward(self, input_tensor):
        
        # Store the activations from input_tensor
        self.activations = input_tensor

        # Apply softmax to the scores: e(x_i) / sum(e(x))

        # subtracting the maximum value before applying the exponential function 
        # helps prevent numerical instability and overflow problems, 
        # making the computation of the SoftMax function more accurate and robust.
        exp_scores = np.exp(input_tensor - np.max(input_tensor, axis= 0, keepdims=True))
        self.activations = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        return self.activations

    def backward(self, label_tensor):
        # Subtracting the one-hot encoded label tensor from 
        # the activations tensor effectively increases the error for the incorrect classes 
        # and leaves the error for the correct class unchanged.
        error_tensor = self.activations - label_tensor

        return error_tensor


    def loss(self, label_tensor):
        
        # Cross entropy loss-
        #      -loss is negative log of the activation of the correct position
        
        # Iterate over all elements of the batch and sum the loss
        loss = -np.sum(label_tensor * np.log(self.activations))

        return loss

