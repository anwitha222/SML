import numpy as np
import pandas as pd

def load_dataset(csv_path):
    #function should:

    #load the dataset from a CSV file
    #Arguments should be the path to the file
    #return X and Y 

    df = pd.read_csv(csv_path, header = None) #read the CSV into a pandas data frame
    y = df.iloc[:, 0].to_numpy().astype(int) #array of shape (n_samples) with vales 0 and 1
    X = df.iloc[:, 1:].to_numpy().astype(float) #array of shape (n_samples, n_features)
    return X, y


def standardize_fit(X_train):
    #standardize the training features:
    #z = (x-mean) / std
    #returns X_scales, which is the standardized version of X-train
    #mean and std values of each feature

    mean = X_train.mean(axis=0) #mean of each column
    std = X_train.std(axis=0, ddof=0) #standard deviation of each column
    std[std == 0.0] = 1.0 #cant divide by 0
    return (X_train - mean) / std, mean, std

def standardize_apply(X, mean, std):
    return(X-mean)/ std

