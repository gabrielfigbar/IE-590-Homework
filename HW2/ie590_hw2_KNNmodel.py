'''
IE590 HW2 KNN model 
Author: Gabriel Figueiredo Barbosa
Date: Oct 10, 2025
Description: This script reads training and testing data from CSV files, 
    implements a KNN model with different distance metrics (L1 and L2) 
    and k-values (1,2, 5, 20) and visualizes the results.
'''

############################################################
# Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################################################
# Function definitions
def readData(fileName):
    '''
    Description:
        Reads data from a CSV file and returns it as a list of tuples (x, y, label).
    Args:
        fileName (str): The name of the CSV file to read.
    Returns:
        list: A list of tuples containing the data points.
    '''
    data = []
    return data

def plotData(data):
    '''
    Description:
        Plots the data points with different colors for different labels.
    Args:
        data (list): A list of tuples containing the data points.
    Returns:
        None
    '''
    return

def KNNmodel(data, metric, k):
    '''
    Description:
        Implements the KNN model using the specified distance metric and k-value.
    Args:
        - data (list): A list of tuples containing the training data points.
        - metric (str): The distance metric to use ('l1' or 'l2').
        - k (int): The number of nearest neighbors to consider.
    Returns:
        list: A list of tuples containing the model data points.
    '''
    model_data = []
    return model_data

############################################################
# Main execution
def main():
    data_test = readData("testKNN.csv")
    data_train = readData("trainKNN.csv")

    plotData(data_test)

    for metric in ['l2', 'l1']:
        for k in [1, 2, 5, 20]:
            model_data = KNNmodel(data_train, metric, k)
            plotData(model_data)

    return

if __name__ == "__main__":
    main()