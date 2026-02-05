import pandas as pd
import numpy as np

def normalize_column(column):
    '''
    args: Dataframe column
    This function takes in the column, gets the min, and max values respectively
    Then normalizes the column 
    '''
    max_val = column.max()
    min_val = column.min()
    normalized = (column - min_val) / (max_val - min_val)
    return normalized

def load_data(path):
    '''
    args: csv file path
    This function reads the csv from path, and apply normalization
    '''
    df = pd.read_csv(path, names=['X', 'Y', 'X_Vel', 'Y_Vel'])
    df_norm = pd.DataFrame(columns = ['X', 'Y', 'X_Vel', 'Y_Vel'])
    for column in df.columns.values:
        df_norm[column] = normalize_column(df[column])
      
    return df_norm
    
def fetch_features(df):
    '''
    args: Takes in dataframe
    This function reads the first two column of dataframe as inputs, and last two as outputs 
    '''
    input_features = np.array(df.iloc[:, :2].values)
    outputs = np.array(df.iloc[:, 2:].values)
    return input_features, outputs


def split_data(df, train_split_ratio=0.7, val_split_ratio=0.15):
    '''
    args: Takes in dataframe, training, validation and testing ratio
    This function takes the dataframe, shuffles the dataframe
    calculate the number of training, and validation ratio
    Then get the train/val/test data and fetch the input, output features of each split
    '''
    df_shuffled = df.sample(frac=1, random_state=0)
    n_samples = len(df_shuffled)
    n_train = int(train_split_ratio * n_samples)
    n_val = int(val_split_ratio * n_samples)
    
    train_data = df_shuffled.iloc[:n_train].reset_index(drop=True)
    validation_data = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_data = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)
    
    X_train, y_train = fetch_features(train_data)
    X_val, y_val = fetch_features(validation_data)
    X_test, y_test = fetch_features(test_data)
    
    return X_train, y_train, X_val, y_val, X_test, y_test