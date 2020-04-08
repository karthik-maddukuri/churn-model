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
from src.data.make_dataset import load_training_data
from src.localpaths import *


#decorator
@click.group() 
def cli():
    pass

@cli.command()
@click.option('--file-name', type=str, required=True)
def predict(file_name):
    """
    
    """
    print(file_name)



if __name__ == "__main__":
    cli()