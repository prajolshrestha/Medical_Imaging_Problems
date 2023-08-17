import numpy as np
from AllFunctions.data import Data
from AllFunctions.irisData import IrisData
from AllFunctions.neuralNetwork import NeuralNetwork
from AllFunctions.inputLayer import InputLayer
from AllFunctions.fullyConnected import FullyConnected
from AllFunctions.reLU import ReLU
from AllFunctions.softMax import SoftMax




def main():  # test_iris_data, TestNeuralNetwork

    # load sample data
    iris_data = IrisData()
    categories = iris_data.categories
    input_size = iris_data.feature_dim

    # set network options
    learning_rate = 1e-4  # set same value for all layers here
    batch_size = iris_data.full_batch  # use all samples for this exercise

    # ---------------------------
    # Construction of the network
    # ---------------------------
    net = NeuralNetwork()

    net.data_layer = InputLayer(iris_data)

    # fully connected layer 1
    fcl_1 = FullyConnected(input_size, categories, batch_size, learning_rate)
    net.layers.append(fcl_1)
    net.layers.append(ReLU(categories, batch_size))

    # fully connected layer 2
    fcl_2 = FullyConnected(categories, categories, batch_size, learning_rate)
    net.layers.append(fcl_2)
    net.layers.append(ReLU(categories, batch_size))

    net.loss_layer = SoftMax(categories, batch_size)

    # -----------------------
    # Training of the network
    # -----------------------
    net.train(2000)

    # ---------------------------
    # Testing/running the network
    # ---------------------------
    # compute results for test data
    data, labels = iris_data.get_test_set()
    results = np.round(net.test(data))

    # ---------------------------
    # Statistics
    # ---------------------------
    # compute accuracy
    accuracy = compute_accuracy(results.T, labels.T)

    # report the result
    if accuracy > 0.9:
        print('\nSuccess!')
    else:
        print('\nFailed! (Network\'s accuracy is below 90%)')
    print('In this run, on the iris dataset, we achieve an accuracy of {} %'.format(str(accuracy * 100)))

    net.show()


def compute_accuracy(results, labels):
    correct = 0
    wrong = 0

    # Compute nr. of correct and wrong prediction
    for column_results, column_labels in zip(results, labels):
        #check if prediction is correct or not
        if column_results[column_labels > 0].all() > 0:
            correct += 1
        else:
            wrong += 1
    
    # Compute accuracy
    if correct == 0 and wrong == 0:
        return 0 # accuracy cannot be computed
    else:
        return correct / (correct + wrong) # Computes accuracy


    # Check if the python script is being run as the main program or if it is being imported as a module into another script.
if __name__ == '__main__':
    main()