# Standard imports
import os
import pickle
import sys
sys.path.append('.') 

#Third-party imports
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

#Local imports
from src.data.make_dataset import load_training_data, clean_X
from src.localpaths import *


#decorator
@click.group() 
def cli():
    pass

@cli.command()
@click.option('--file-name', type=str, required=True)
def predict(file_name):
    """
    Predicts Churn or not for all the data in file_name.
    file_name must be a csv and order of the column names must be
    similar to that of the X_train
    """
    # load data
    X = pd.read_csv(file_name)
    # clean and featurize data
    X = clean_X(X)
    # make predictions

    # print predictions
    print(X)


if __name__ == "__main__":
    cli()