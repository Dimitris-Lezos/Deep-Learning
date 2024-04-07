"""
This class represents the full Neural Network
"""
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from functions import activation_functions, loss_functions
from Layer import Layer
from typing import List
import json


# control debug output
verbose = False

class ANN:
    """
    This class represents the whole Neural Network.
    It contains a list of Layers
    """

    def __init__(self, configuration: {}):
        """
        Initialization of ANN
        :param configuration: The configuration read from a JSON. First level and 'adam' are parsed here, 'layers' are passed at ANN.createANN
        """
        # Create the ANN's layers
        self.layers = ANN.createANN(self, x_train.shape[1], configuration['ANN']['layers'])
        # Read the training parameters
        self.loss_function = loss_functions[configuration['ANN']['loss_function']]
        self.batch_size = configuration['ANN']['batch_size']
        self.epochs = configuration['ANN']['epochs']
        self.tol = configuration['ANN']['tol']
        # Read the Adam parameters
        self.a = configuration['ANN']['adam']['a']
        self.b1 = configuration['ANN']['adam']['b1']
        self.b2 = configuration['ANN']['adam']['b2']
        self.e = configuration['ANN']['adam']['e']
        self.eval_ = {
            'cost': list(),
            'train_acc': list(),
            'valid_acc': list()
        }

    def adam(self,
             layer: Layer,
             N=1,
             b1t=float(0.9),
             b2t=float(0.999),
             ) -> None:
        """
        Implements the adam algorithm
        :param layer: The layer of the NN to calculate, the learn() method will be called and the weights will be updated
        :param N: The batch size
        :param a: The a parameter of Adam
        :param b1: The b1 parameter of Adam
        :param b1t: The b1^t parameter of Adam
        :param b2: The b2 parameter of Adam
        :param b2t: The b2^t parameter of Adam
        :param e: The e parameter of Adam
        :return: None
        """
        g = layer.d / N
        layer.m = self.b1 * layer.m + (1 - self.b1) * g
        layer.v = self.b2 * layer.v + (1 - self.b2) * np.power(g, 2)
        m = layer.m / (1 - b1t)
        v = layer.v / (1 - b2t)
        dw = -1 * (self.a * m / (np.sqrt(v) + self.e))
        layer.learn(dw)


    def _compute_cost(self, target, output):
        """
        Compute cost function.
        Parameters:
            y_enc : array, shape = (n_samples, n_labels) one-hot encoded class labels.
            output : array, shape = [n_samples, n_output_units] Activation of the output layer (forward propagation)
        Returns:
            cost : float Regularized cost
        """
        #L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -target * (np.log(output))
        term2 = (1. - target) * np.log(1. - output)
        cost = np.sum(term1 - term2)
        return cost

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> None:
        """
        Learn weights from training data.
        :param x_train: array, shape = [n_samples, n_features] Input layer with original features.
        :param y_train: array, shape = [n_samples] Target class labels.
        :param x_valid: array, shape = [n_samples, n_features] Sample features for validation during training
        :param y_valid: array, shape = [n_samples] Sample labels for validation during training
        :return: None
        """
        # Run epochs times
        for epoch in range(self.epochs):
            # TODO: pass the whole batch as a matrix
            indexes = list(range(len(x_train)))
            random.shuffle(indexes)
            i = 0
            batch_loss = 0.0
            b1t = self.b1
            b2t = self.b2
            # Run all the training, every batch_size update the weights, if tolerance exceeded stop training
            for x in indexes:
                # Feed Forward
                in_x = x_train[x]
                target = y_train[x]
                for layer in self.layers:
                    in_x = layer.eval(in_x)
                # Calculate loss
                _, d_loss, _ = self.loss_function(target, in_x)
                if i == 0 and verbose:
                    print(target, in_x, d_loss) #, self.layers[0].weights)
                # Back Propagation
                for layer in reversed(self.layers):
                    d_loss = layer.back(d_loss)
                i += 1
                # When we have evaluated the batch, do the training
                # Equivalent to (i % self.batch_size == 0) where i is always increased
                if i == self.batch_size:
                    i = 0
                    # Update weights
                    for layer in self.layers:
                        self.adam(layer, N=self.batch_size, b1t=b1t, b2t=b2t)
                    # Calculate b1^t, b2^t for the next t
                    b1t = b1t * self.b1
                    b2t = b2t * self.b2
            # Evaluate end conditions after each epoch, this deviates from the Adam algorithm which evaluates
            # end conditions every time it runs
            y_pred = self.predict(x_valid)
            # Get the loss, this will also assign to y_pred the probabilities from sofmax,
            # if the loss is CCE
            loss, _, y_pred = self.loss_function(y_valid, y_pred)
            # Let's open a parenthesis here (
            try:
                # The following output has been copied from Lab-4/mlp.py
                # It seems to work only on classification tasks
                cost = self._compute_cost(y_valid, y_pred)
                y_train_pred = self.predict(x_train)
                _, _, y_train_pred = self.loss_function(y_train, y_train_pred)
                y_valid_pred = self.predict(x_valid)
                _, _, y_valid_pred = self.loss_function(y_valid, y_valid_pred)
                train_acc = ((np.sum(y_train == y_train_pred)).astype(float) / x_train.shape[0])
                valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(float) / x_valid.shape[0])
                print('Epoch %d/%d | Cost: %.2f ''| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                                 (epoch+1, self.epochs, cost, train_acc * 100, valid_acc * 100))
                self.eval_['cost'].append(cost)
                self.eval_['train_acc'].append(train_acc)
                self.eval_['valid_acc'].append(valid_acc)
            except Exception as e:
                print(e)
            # ) close the parenthesis end continue our own loss report
            print(f'Epoch {epoch+1} Loss: {np.sum(loss) / y_pred.shape[0]}')
            # Get the mean loss for all outputs (not the sum of losses of all outputs
            # Check end conditions
            if np.sum(loss) / y_pred.shape[0] < self.tol:
                print(f'Loss minimized in {epoch} epochs!')
                break
        # Report final findings
        # Get predictions on the train and valid set
        y_train_pred = self.predict(x_train)
        # Call loss_function to apply any needed transformation (e.g. softmax)
        _, _, y_train_pred = self.loss_function(y_train, y_train_pred)
        y_valid_pred = self.predict(x_valid)
        # Call loss_function to apply any needed transformation (e.g. softmax)
        _, _, y_valid_pred = self.loss_function(y_valid, y_valid_pred)
        try:
            confsusion_matrix_train = confusion_matrix(y_train, y_train_pred, labels=None)
            confusion_matrix_valid = confusion_matrix(y_valid, y_valid_pred, labels=None)
            print('\n Conf matrix, Train Set, Neural Net')
            print(confsusion_matrix_train)
            print('\n Conf matrix, Validation Set, Neural Net')
            print(confusion_matrix_valid)
            # Measures of performance: Precision, Recall, F1
            print('Neural Net: Macro Precision, recall, f1-score')
            print(precision_recall_fscore_support(y_valid, y_valid_pred, average='macro'))
            print('Neural Net: Micro Precision, recall, f1-score')
            print(precision_recall_fscore_support(y_valid, y_valid_pred, average='micro'))
        except Exception as e:
            print(e)
        print('######################################################################')

    def predict(self, X_valid: np.ndarray) -> np.ndarray:
        """
        Gives the output of the trained Neural Network
        :param X_valid: The inputs to predict on
        :return: An array with the outputs, one row for each row provided in X_valid
        """
        predictions = list()
        for in_x in X_valid:
            for layer in self.layers:
                in_x = layer.eval(in_x)
            predictions.append(in_x)
        return np.asarray(predictions)


    def createANN(self, inputs: int, layers_descriptor: np.array) -> List[Layer]:
        """
        Creates an ANN based on the layers_descriptor with the provided number of inputs
        :param inputs: The number of inputs to the initial layer
        :param layers_descriptor: The details of the layers coming from the configuration file
        :return: A list containing each layer starting from the one that is fed the inputs
        """
        layers = []
        for layer_description in layers_descriptor:
            nodes = layer_description['nodes']
            activation = 'relu'
            if 'activation' in layer_description:
                activation = layer_description['activation']
            input_weights = None
            if 'input_weights' in layer_description:
                input_weights = layer_description['input_weights']
            biases = None
            if 'biases' in layer_description:
                biases = layer_description['biases']
            layers.append(
                Layer(
                    inputs,
                    nodes,
                    input_weights=input_weights,
                    biases=biases,
                    activation=activation_functions[activation]
                )
            )
            inputs = nodes
        return layers


def read_configuration(config_filename='config.json') -> {}:
    """
    Reads the JSON configuration file
    :param config_filename: The name of the configuration file, default 'config.json'
    :return: A dictionary containing the configuration file (JSON -> {})
    """
    try:
        return json.load(open(config_filename))
    except Exception as ex:
        print("Failed to read configutation from", config_filename, 'with error', ex)
    return {}


if __name__ == '__main__':
    # Read the configuration file
    config_filename = 'config.json'
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    configuration = read_configuration(config_filename)
    # Read train and test data
    train_data = pd.read_csv(configuration['train_data_filename'], header=configuration['header'], dtype=float, skiprows=configuration['train_data_skiprows'])
    test_data = pd.DataFrame()
    try:
        test_data = pd.read_csv(configuration['test_data_filename'], header=configuration['header'], dtype=float, skiprows=configuration['test_data_skiprows'])
    except Exception as e:
        print('Failed to read test data: ', e)
        print('Failed to read test data, will split train_data instead')
    output_size = configuration['ANN']['layers'][len(configuration['ANN']['layers'])-1]['nodes']
    random_state = configuration['random_state']
    if len(test_data.columns) == 0 or len(train_data.columns) == len(test_data.columns)+output_size:
        # train_data_filename contains train and validation data
        x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:,:-output_size].to_numpy(),
                                                            train_data.iloc[:,-output_size:].to_numpy(),
                                                            test_size=configuration['test_size'],
                                                            random_state=random_state)
    elif len(train_data.columns) == len(test_data.columns):
        # train_data_filename contains train data
        # test_data_filename contains test data
        # no validation data provided we will steal a few from the test ones
        x_train = train_data.iloc[:,:-output_size].to_numpy()
        y_train = train_data.iloc[:,-output_size:].to_numpy()
        x_test = test_data.iloc[:,:-output_size].to_numpy()
        y_test = test_data.iloc[:,-output_size:].to_numpy()
        test_data = None
    else:
        # We don't know how to interpret the provided files between train, test and verification
        print('Train data (with target column) from file', configuration['train_data_filename'],
              'and Test data (with target column) from file', configuration['test_data_filename'],
              'have different number of columns!')
        print('Aborting!')
        exit(-1)
    # Create the ANN
    ann = ANN(configuration)
    # Run the ANN training, end report results
    ann.fit(x_train, y_train, x_test, y_test)
    # Generate Predictions for Test data
    if test_data is not None:
        # x_test = test_data.to_numpy()
        y_pred = ann.predict(test_data.to_numpy())
        # Append predictions to test_data and save in output file
        test_data[train_data.columns[-output_size:]] = y_pred
        # c = 0
        # for column in train_data.columns[-output_size:]:
        #     test_data.loc[:, column] = y_pred[:,c]
        #     c += 1
        test_data.to_csv('prediction.csv', index=False)
        print(f'Predictions for "{configuration["test_data_filename"]}" file saved to "prediction.csv"')
