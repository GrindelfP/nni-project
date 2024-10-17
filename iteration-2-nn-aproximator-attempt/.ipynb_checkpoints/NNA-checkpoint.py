# TODO: add dataset generators here

import numpy as np

from tqdm import tqdm
from prettytable import PrettyTable
from typing import Callable, List, Tuple

class Approximator:

    
    def _generate_weights(self) -> list[list[float]]:
        """
            Generates matrix of weights.
            @return: returns matrix of random weights.
        """

        weights = [
            np.random.uniform(-1.0, 1.0, (self.input_layer_size + 1, self.hidden_layer_size)),
            np.random.uniform(-1.0, 1.0, (self.hidden_layer_size + 1, 1))
        ]

        return weights

    
    def _generate_neurons(self) -> list[list[float]]:
        """
            Generates matrix of neurons.
            @return: returns matrix of neurons.
        """

        matrix = [
            np.zeros(self.input_layer_size + 1),
            np.zeros(self.hidden_layer_size + 1),
            np.zeros(1)                     
        ]
        
        # Set the last elements of the first and second vectors to 1
        matrix[0][-1] = 1
        matrix[1][-1] = 1

        return matrix

    
    def __init__(
        self, 
        input_size=1, 
        training_set_size=1000,
        k1=1,
        k2=1
    ):
        """
            Constructs an instance of NeuralNetworkApproximator.
            @param input_size: dimesions count of function to approximate. Default value 
            @param training_set_size: number of trainig samples, which influenses hidden layer size.
            @param k1: tunable parameter, which influenses hidden layer size.
            @param k2: tunable parameter, which influenses hidden layer size.
        """
        
        self.input_layer_size = input_size
        self.hidden_layer_size = int(np.log10(training_set_size)**(-k1) * (k2 * training_set_size/(input_size + 2)))
        self.output_layer_size = 1
        self.weights = self._generate_weights()
        self.neurons = self._generate_neurons()
        

    def _sigmoid(self, z) -> None:
        """
            Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    
    def _sigmoid_derivative(self, z) -> None:
        """
            Derivative of the sigmoid function.
        """
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    
    def _linear(self, z):
        return z  # Линейная активация для выхода

    
    def _linear_derivative(self, z):
        return 1  # Производная линейной функции — это просто 1

    
    def train(self, data, epochs_num=1000, lm_param=0.00001) -> None:
        """
            Trains the neural network using Levenberg-Marquardt algorithm.
            
            @param data: List of tuples (x, y) representing training samples.
            @param epochs_num: Number of epochs for the training process.
            @param lm_param: Initial Levenberg-Marquardt parameter (lambda).
        """
        for epoch in tqdm(range(epochs_num), desc="Training", unit="epoch"):
            mse = 0  # Mean squared error for the epoch

            for x, y in data:
                # Forward propagation (x already includes bias as the last element)
                x_with_bias = np.append(x, 1)  # Добавляем bias к x
                hidden_input = np.dot(x_with_bias, self.weights[0])
                hidden_output = np.append(self._sigmoid(hidden_input), 1)  # hidden_output with bias

                final_input = np.dot(hidden_output, self.weights[1])
                final_output = final_input # self._linear(final_input)

                # Compute the error (Mean Squared Error)
                error = y - final_output
                mse += error**2

                # Backpropagation
                output_delta = error * final_input # self._linear_derivative(final_input)
                hidden_delta = output_delta.dot(self.weights[1][:-1].T) * self._sigmoid_derivative(hidden_input)

                # Update weights using LM algorithm
                self.weights[1] += lm_param * np.outer(hidden_output, output_delta)
                self.weights[0] += lm_param * np.outer(x_with_bias, hidden_delta)

            mse /= len(data)  # Средняя ошибка для эпохи

            # Adjust lambda (LM parameter) adaptively
            if mse < 1e-6:
                break
            elif epoch % 10 == 0:
                lm_param *= 0.9

        return mse

    
    def predict(self, x) -> None:
        """
            Predicts the output for a given input vector x.
            
            @param x: Input vector for which to predict the output.
            @return: Predicted output value.
        """
        x_with_bias = np.append(x, 1)  # Добавляем bias к x
        hidden_input = np.dot(x_with_bias, self.weights[0])
        hidden_output = np.append(self._sigmoid(hidden_input), 1)  # hidden_output with bias
        final_input = np.dot(hidden_output, self.weights[1])
        final_output = self._sigmoid(final_input)
        return final_output

    
    def test(self, test_data) -> None:
        """
            Tests the neural network on a given dataset and computes the mean squared error.
            
            @param test_data: List of tuples (x, y) representing test samples.
            @return: Mean squared error of predictions.
        """
        total_error = 0
        for x, y in test_data:
            prediction = self.predict(x)
            error = y - prediction
            total_error += error**2
            
        mse = total_error / len(test_data)
        return mse

    
    def download_weights_biases(self) -> tuple[list, list, list, list]:
        """
            Returns a list of weights and biases for each layer in the neural network.
            Biases are treated as the last weights for each layer.
            
            @return: A list containing weights and biases for each layer.
        """
        first_array = self.weights[0][:-1]
        second_array = self.weights[0][-1]
        third_array = self.weights[1][:-1]
        fourth_array = self.weights[1][-1]         
        
        return first_array, second_array, third_array, fourth_array

    
    def info(self) -> None:
        """
            Displays the architecture of the neural network in a formatted table.
        
            This method prints a table showing the different layers of the neural network
            along with the number of parameters for each layer. The table includes:
            - Input Layer: Number of input parameters (excluding the bias neuron).
            - Hidden Layer: Number of parameters in the hidden layer (including the bias).
            - Output Layer: Number of parameters in the output layer (including the bias).
            - Total Parameters Count: The sum of all parameters across the layers.
        
            @returns: None
        """
        table = PrettyTable()
        
        # Устанавливаем заголовок таблицы
        table.title = "NEURAL NETWORK APPROXIMATOR"
        table.field_names = ["Layer", "Number of Parameters"]
        
        # Входной слой
        input_params = self.input_layer_size  # Параметры входного слоя (без учета смещения)
        table.add_row(["Input Layer", input_params])
        
        # Скрытый слой
        hidden_params = self.hidden_layer_size + 1  # Параметры скрытого слоя (включая смещение)
        table.add_row(["Hidden Layer", hidden_params])
        
        # Выходной слой
        output_params = self.output_layer_size + 1  # Параметры выходного слоя (включая смещение)
        table.add_row(["Output Layer", output_params])
        
        # Общее количество параметров
        total_params = input_params + hidden_params + output_params
        
        print(table)
        print(f"Total Parameters Count: {total_params}")
