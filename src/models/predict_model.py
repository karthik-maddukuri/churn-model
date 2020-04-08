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
from  src.features.build_features import featurize_X
from src.localpaths import *
from src.models.train_model import load_pickled_model


PICKLED_MODEL_FILENAME = '1017545180110469376.pkl'



#decorator
@click.group() 
def cli():
    pass


def predict(file_name, proba=False):
    """
    Predicts Churn or not for all the data in file_name.
    file_name must be a csv and order of the column names must be
    similar to that of the X_train
    """
    # load data
    X = pd.read_csv(file_name)

    # clean and featurize data
    X = clean_X(X)
    X = featurize_X(X, predict=True)

    # Load model
    model = load_pickled_model(PICKLED_MODEL_FILENAME)

    # make predictions
    if proba:
        predictions = model.predict_proba(X)[:, 1]
    else:
        predictions = model.predict(X)

    # print predictions
    return predictions


@cli.command()
@click.option('--file-name', type=str, required=True)
def click_predict(file_name):
    """
    Predicts Churn or not for all the data in file_name.
    file_name must be a csv and order of the column names must be
    similar to that of the X_train
    """
    
    predictions = predict(file_name)
    print(predictions)





if __name__ == "__main__":
    cli()