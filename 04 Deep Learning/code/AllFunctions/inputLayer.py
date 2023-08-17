import numpy as np
from AllFunctions.data import Data

# InputLayer
class InputLayer:

    # Constructor with input: type annotation (data: Data)
    # Type Annotations provide additional information about the expected types of functions or method parameters,
    # which can be helpful for both code readability and static type checking by tools like type checkers and linters.
    def __init__(self,data: Data): # data is an instance of class Data
        self.input_tensor, self.label_tensor = data.get_train_set()

    def forward(self):
        return np.copy(self.input_tensor) # np.copy() ensures that the original data is not modified during further operations
    
    def backward(self):
        return np.copy(self.label_tensor)