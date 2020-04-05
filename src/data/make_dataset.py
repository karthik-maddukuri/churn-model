# Standard imports
import os
import sys
sys.path.append('.') 

#Third-party imports
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Local imports
from src.localpaths import *

#decorator
@click.group() 
def cli():
    pass


@cli.command()
def create_train_test_split():
    """ Splitting the data using holdout method 
    in the data/raw dir 
    """
    print('Loading the data')
    df = pd.read_csv(RAW_DATA_PATH)
    print('Splitting X and y for training and testing')
    X = df.drop(columns=['Churn'])
    y = df[['Churn']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

    print('Saving files')
    X_train.to_csv(X_TRAIN_RAW_PATH, index = False)
    X_test.to_csv(X_TEST_RAW_PATH, index = False)
    y_train.to_csv(Y_TRAIN_RAW_PATH, index = False)
    y_test.to_csv(Y_TEST_RAW_PATH, index = False)


@cli.command()
def create_clean_training_data():
    """
    Reads in the raw X and y trainging data, cleans it and writes the clean
    training data out to the data/interim directory
    """
    print('loading data')
    X_train, y_train = load_training_data()

    print('cleaning data')
    bad_values_idxs = X_train[X_train['TotalCharges'] == ' '].index
    X_train.loc[bad_values_idxs, 'TotalCharges'] = 20
    X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)

    print('Writing data')
    X_train.to_csv(X_TRAIN_CLEAN_PATH, index = False)
    y_train.to_csv(Y_TRAIN_CLEAN_PATH, index = False)








# Helper function for loading training and testing data, instead of using the localpaths functions everytime
def load_training_data(clean = False, final=False):
    """ Return the X_train and y_train data if they exist
    """
    if clean:
        X_train = pd.read_csv(X_TRAIN_CLEAN_PATH)
        y_train = pd.read_csv(Y_TRAIN_CLEAN_PATH)
    elif final:
        X_train = pd.read_csv(X_TRAIN_FEATURIZED_PATH)
        y_train = pd.read_csv(Y_TRAIN_FEATURIZED_PATH)
    else:
        X_train = pd.read_csv(X_TRAIN_RAW_PATH)
        y_train = pd.read_csv(Y_TRAIN_RAW_PATH)
    
    return X_train, y_train


def load_test_data():
    """ 
    Return the X_test and y_test data if they exist
    """
    X_test = pd.read_csv(X_TEST_RAW_PATH)
    y_test = pd.read_csv(Y_TEST_RAW_PATH)
    
    return X_test, y_test
    



if __name__ == "__main__":
    cli()