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
from pathlib import Path
from statistics import mode

############################################################
# Function definitions
def readData(fileName):
    '''
    Description:
        Reads data from a CSV file and returns it as a pandas dataframe.
        Column headers are set to 'x', 'y', and 'label'; this means that
        test data will have NaN values in the 'label' column.
    Args:
        fileName (str): The name of the CSV file to read.
    Returns:
        df (DataFrame): A pandas dataframe containing the data.
    '''
    # Define path for data folder
    data_path = Path(__file__).parent / "data"
    # Set column names dataframe
    col_names = ['x', 'y', 'label']
    # Read CSV files into DataFrame
    df = pd.read_csv(data_path / fileName, header=None, names = col_names)
    return df

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

def KNNmodel(df_test, df_train, metric, k):
    '''
    Description:
        Implements the KNN model using the specified distance metric and k-value.
    Args:
        - df_test (DataFrame): the testing data as a pandas dataframe
        - df_train (DataFrame): The training data as a pandas dataframe.
        - metric (str): The distance metric to use ('l1' or 'l2').
        - k (int): The number of nearest neighbors to consider.
    Returns:
        df_model (DataFrame): The dataframe with predicted labels for test data.
    '''
    # Create a copy of the dataframe to avoid modifying the original data
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    for row_test in df_test.itertuples(index=False, name=None):
        # Iterates through all the rows in test data; we will compare them to train data
        min_distances = [0] * k # k-sized array to store the k smallest distances
        neighbor_labels = [0] * k # k-sized array to store the neighbors' label

        for row_train in df_train.itertuples(index=False, name=None):
        # Iterates through all the rows in train data, comparing test rows to them
            farthest_dist = max(min_distances) # Farthest distance
            if metric == 'l1': # Uses l1 metric for distance calculation on this datapoint
                dist = abs(row_train[0] - row_test[0]) + abs(row_train[1] - row_test[1])
            elif metric == 'l2': # Uses l2 metric for distance calculation on this datapoint
                dist = ((row_train[0] - row_test[0])**2 + (row_train[1] - row_test[1])**2)**0.5

            if dist < farthest_dist: # In case this datapoint is a current k-closest neighbor
                min_distances[min_distances.index(farthest_dist)] = dist # Substitute farthest neighbor
                neighbor_labels[min_distances.index(farthest_dist)] = row_train[2] # Substitute farthest neighbor label
        # After iterating through all train data, we have the k closest neighbors
        # Now we need to determine the most common label among them
        predicted_label = mode(neighbor_labels)
        # FIND WAY TO INPUT THIS INTO THE TEST DATAFRAME (COL 'label', WHICH IS CURRENTLY NAN)
    return df_test

############################################################
# Main execution
def main():
    df_test = readData("testKNN.csv")
    df_train = readData("trainKNN.csv")

    df_model = KNNmodel(df_test, df_train, 'l2', 5)
    '''
    for metric in ['l2', 'l1']:
        for k in [1, 2, 5, 20]:
            model_data = KNNmodel(data_train, metric, k)
            plotData(model_data)
    '''
    return

if __name__ == "__main__":
    main()