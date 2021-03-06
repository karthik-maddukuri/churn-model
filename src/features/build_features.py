# Standard imports
import os
import pickle
import sys
sys.path.append('.') 

#Third-party imports
import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Local imports
from src.localpaths import *
from src.data.make_dataset import load_training_data

#decorator
@click.group() 
def cli():
    pass


def featurize_X(X, predict=False):
    """
    Applies featurization to X_train only
    """
    X = drop_customer_id(X)
    X = transform_binary_categorical(X)
    X = one_hot_encode_categorical_features(X, predict=predict)
    X = drop_high_vif_features(X, predict=predict)

    return X



@cli.command()
def create_featurized_data():
    """
    This creates x and y training files ready for modeling
    saves the data to data/processed
    """
    print('loading data ')
    X_train, y_train = load_training_data(clean=True)

    print('Featurizing data')
    X_train = featurize_X(X_train)
    y_train = transform_target(y_train)

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



def one_hot_encode_categorical_features(X_train, save_encoder=True, predict=False):
    """
    This one hot encodes the categorical features, add these to X-train,
    then drops the original columsn. Returns the transformed X_train data as dataframe
    """
    ohe_filepath = os.path.join(SRC_FEATURES_DIRECTORY, 'ohe-hot-encoder.pkl')
    cols_to_one_hot_encode = X_train.dtypes[X_train.dtypes == 'object'].index

    if predict:
        with open(ohe_filepath, 'rb') as f:
            ohe = pickle.load(f)

    else:
        ohe = OneHotEncoder(drop='first',sparse=False)
        ohe.fit(X_train[cols_to_one_hot_encode])

    ohe_features = ohe.transform(X_train[cols_to_one_hot_encode])
    ohe_feature_names = ohe.get_feature_names(cols_to_one_hot_encode)

    ohe_df = pd.DataFrame(ohe_features, columns = ohe_feature_names)

    X_train = X_train.assign(**ohe_df)
    X_train = X_train.drop(columns=cols_to_one_hot_encode )

    if save_encoder and not predict:
        ohe_filepath = os.path.join(SRC_FEATURES_DIRECTORY, 'ohe-hot-encoder.pkl')
        print('pickling one-hot-encoder')
        with open(ohe_filepath, 'wb') as f:
            pickle.dump(ohe, f)

    return X_train


def drop_high_vif_features(X_train, predict=False):
    """
    Drops features with a variance inflation factor greater than 10
    """
    dropped_cols_pickle_filepath = os.path.join(SRC_FEATURES_DIRECTORY, 'cols-to-drop.pkl')
    if predict:
        with open(dropped_cols_pickle_filepath, 'rb') as f:
            cols_to_drop = pickle.load(f)
        X_train = X_train.drop(columns=cols_to_drop)

    else:
        finished = False
        cols_to_drop = []
        while not finished:
            vifs = [variance_inflation_factor(X_train.values,i) for i in range(len(X_train.columns))]
            high_vifs = sorted(zip(X_train.columns, vifs), key=lambda x: x[1], reverse=True)
            high_vif_col, high_vif_value = high_vifs[0]
            if high_vif_value >= 10:
                print(f'Dropping column {high_vif_col} as it has {high_vif_value:.1f} >=10')
                X_train = X_train.drop(columns=[high_vif_col])
                cols_to_drop.append(high_vif_col)
            else:
                print('finished dropping columns')
                finished = True

        with open(dropped_cols_pickle_filepath, 'wb') as f:
            pickle.dump(cols_to_drop, f)
     
    return X_train





def transform_target(y_train):
    """
    trasnform churn to 0s and 1s for modeling
    """
    y_train['Churn'] = y_train['Churn'].map({'Yes': 1, 'No': 0})
    
    return y_train




if __name__ == "__main__":
    cli()