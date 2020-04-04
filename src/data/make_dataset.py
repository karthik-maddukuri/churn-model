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

@click.group() #decorator
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


if __name__ == "__main__":
    cli()