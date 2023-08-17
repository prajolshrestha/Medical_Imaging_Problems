import numpy as np

# Base class: 'Data'
# It defines the fundamental structure and behaviour of data storage
# and retrieval methods.
# class Data:

#     def __init__(self, inputs, labels):

#         # Create a dictonary for train and test sets (each with two sub dictonaries - inout and label)
#         self.data = {'train':
#                      {'input': np.array(inputs[0]),
#                       'label': np.array(labels[0])},
#                      'test':
#                      {'input': np.array(inputs[1]),
#                       'label': np.array(labels[1])}
#                      }
#         self.n_train = len(inputs[0]) # computes nr. of training samples
#         self.n_test = len(inputs[1]) # Computes nr. of test samples
    
#     # Return transposed input and label data arrays of training sets
#     def get_train_set(self):
#         return self.data['train']['input'].T, self.data['train']['label'].T
    
#     # Return transposed input and label data arrays of test sets
#     def get_test_set(self):
#         return self.data['test']['input'].T, self.data['test']['label'].T


class Data:

    def __init__(self, inputs, labels):

        self.data = {'train':
                     {'input': np.array(inputs[0]),
                      'label': np.array(labels[0])},
                     'test':
                     {'input': np.array(inputs[1]),
                      'label': np.array(labels[1])}
                     }
        self.n_train = len(inputs[0])
        self.n_test = len(inputs[1])

    def get_train_set(self):
        return self.data['train']['input'].T, self.data['train']['label'].T

    def get_test_set(self):
        return self.data['test']['input'].T, self.data['test']['label'].T
