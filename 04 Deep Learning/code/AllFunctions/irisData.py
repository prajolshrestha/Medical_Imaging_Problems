import numpy as np
from sklearn.datasets import load_iris
from random import shuffle

from AllFunctions.data import Data
# Derived Class: 'IrisData'
# Inherits all the methods and attributes defined in 'Data' class.
# In addition, 'IrisData' Class can be extended and customize the behavior of 'Data' class
# by adding its own methods and attributes.

# Here, 'IrisData' class handles iris dataset.
# - it overwrites the constructor of base class to provide data processing logic
# that is specific to iris dataset.
class IrisData(Data):

    train_per_test = 2 # used to determine ratio of train to test data
    categories = 3 # number of classes in the dataset

    def __init__(self, do_shuffle=True): # Defines the constructor
        
        
        inputs = [[],[]] # Initialize "list of empty lists" to store train and test data
        labels = [[],[]]
        samples_by_category = [[] for _ in range(self.categories)] # Initialize "list of empty lists", where each inner list will store data sample from specific category.

        r = IrisData.train_per_test / (IrisData.train_per_test +1) # Ratio of training samples to total samples(used split betn train and test)

        # load data
        iris = load_iris()
        data = iris.data # fetch data(NxD) from iris
        target = iris.target # fetch target from iris

        self.total = data.shape[0] # compute total nr. of samples
        self.feature_dim = data.shape[1] # computes feature dimension
        self.full_batch = 0 # Initialize full_batch (it is used to keep track of total nr. of samples in training set)

         # Organize data based on target by catagory
        for i in range(self.total): # loop through each sample in the dataset
            samples_by_category[target[i]].append(data[i,:]) # Organize data samples by category(ie, append 'current data sample' to appropriate 'category list' based on 'target label')

         # Train test split for each category
        for i in range(self.categories): # loops through each category
            
             # shuffle 
            if do_shuffle: 
                samples_by_category[i] = IrisData._get_shuffled_data(samples_by_category[i]) # calls static method

            n = len(samples_by_category[i]) # computes number of samples for current category
            s = round(r * n) # computes nr. of samples to include in training set 's' , based on ratio 'r'

             # Split train and test data
            inputs[0] += samples_by_category[i][0:s] #Training data
            inputs[1] += samples_by_category[i][s:] #Test data

             # Split train and test labels with one-hot encoding(ie, correct class = 1 & others = 0)
            labels[0] += [[int(j == i) for j in range(self.categories)]] * s # one-hot encoded training labels
            labels[1] += [[int(j == i) for j in range(self.categories)]] * (n-s) # one-hot encoded test labels

            self.full_batch += s # fullbatch = nr. of samples in training set for current category

        # Call constructor of parent class 'Data'
        super().__init__(inputs, labels)

    @staticmethod
    def _get_shuffled_data(data_as_list):
         # shuffles a list of data using Fischer Yates shuffle algo
        index_list = list(range(len(data_as_list)))
        shuffle(index_list)
        return [data_as_list[i] for i in index_list]

