import numpy as np

class ReLU:
    def __init__(self, input_size, batch_size):
        # Store current activations - later used during backpropagation
        self.activations = np.zeros((input_size, batch_size))#pre-allocation

    def forward(self,input_tensor):
        # Store the activations from the input_tensor
        self.activations = input_tensor

        # Apply reLu
        layer_output = np.maximum(0, input_tensor)
        return layer_output

    def backward(self,error_tensor):
        # Gradient is zero whenever the activation is negative
        
        # ReLU has a gradient of 1 for positive activations and
        #  0 for negative activations, 
        # we can use a simple masking operation to set the gradient to 0 
        # where the activations were negative during the forward pass.

        # where(condition, [x, y])
        # Return elements chosen from x or y depending on condition.
        gradient = np.where(self.activations > 0, error_tensor, 0)
        return gradient