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


def store_model_and_results(model, X_train, y_train):
    """
    This saves the model evaluation metrics to /models/model_results.csv
    and saves the pickled models to /models
    """
    model_results_filepath = os.path.join(MODELS_DIRECTORY, 'model_results.csv')
    model_filename = str(hash(np.random.rand())) + '.pkl'
    model_string = str(model)

    accuracy = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='accuracy'))
    precision = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='precision'))
    recall = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='recall'))
    f1 = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='f1'))
    roc_auc = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='roc_auc'))

    data_to_save = {
    'model_filename': [model_filename],
    'model_string': [model_string],
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'f1': [f1],
    'roc_auc': [roc_auc]
    }


    print('Fitting model before pickling')
    model.fit(X_train, y_train)

    print(f'saving the pickled model to {model_filename}')
    with open(os.path.join(MODELS_DIRECTORY, model_filename), 'wb') as f:
        pickle.dump(model,f)
    
    if os.path.exists(model_results_filepath):
        df_results = pd.read_csv(model_results_filepath)
        print('Writing model results to existing results csv file')
        new_results = pd.DataFrame(data_to_save)
        df_results = df_results.append(new_results, ignore_index=True)
    else:
        print('Model results file does not exist')
        print('Creating new model results csv file and writing results')
        df_results = pd.DataFrame(data_to_save)
    df_results.to_csv(model_results_filepath, index=False)



def print_model_results(model, X_train, y_train):
    """
    This prints the model evaluation metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        metric_value = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring=metric))
        print(f'{metric}: {metric_value:.2f}')


def load_model_results():
    """
    Returns the pandas dataframe of model results from /models/model_results.csv
    """
    
    model_results_filepath = os.path.join(MODELS_DIRECTORY, 'model_results.csv')
    df_results = pd.read_csv(model_results_filepath)
    return df_results

def load_pickled_model(model_filename):
    """
    Retuns the unpickeld model when we give the name of the pickled model
    """
    with open(os.path.join(MODELS_DIRECTORY, model_filename), 'rb') as f:
        model = pickle.load(f)

    return model


if __name__ == "__main__":
    pass