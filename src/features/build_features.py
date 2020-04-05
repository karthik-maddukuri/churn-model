# Standard imports
import os
import sys
sys.path.append('.') 

#Third-party imports
import click
import numpy as np
import pandas as pd

#Local imports
from src.localpaths import *
from src.data.make_dataset import load_training_data

#decorator
@click.group() 
def cli():
    pass

@cli.command()
def create_featurized_data():
    """
    This creates x and y training files ready for modeling
    saves the data to data/processed
    """
    print('loading data ')
    X_train, y_train = load_training_data(clean=True)

    print('Featurizing data')
    X_train = drop_customer_id(X_train)
    X_train = transform_binary_categorical(X_train)

    print('Saving data') 
    X_train.to_csv(X_TRAIN_FEATURIZED_PATH, index = False) 
    y_train.to_csv(Y_TRAIN_FEATURIZED_PATH, index = False) 



def drop_customer_id(X_train):
    """
    This drops the cusotmer id column from X_train
    """
    X_train = X_train.drop(columns=['customerID'])

    return X_train


def transform_binary_categorical(X_train):
    """
    Change binary categorical features to 0s and 1s
    """
    X_train['gender'] = X_train['gender'].map({'Female': 1, 'Male': 0})
    X_train['Partner'] = X_train['Partner'].map({'Yes': 1, 'No': 0})
    X_train['Dependents'] = X_train['Dependents'].map({'Yes': 1, 'No': 0})
    X_train['PhoneService'] = X_train['PhoneService'].map({'Yes': 1, 'No': 0})
    X_train['PaperlessBilling'] = X_train['PaperlessBilling'].map({'Yes': 1, 'No': 0})

    return X_train






if __name__ == "__main__":
    cli()