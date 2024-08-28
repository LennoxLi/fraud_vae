# load data 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

def process(data_path="creditcard.csv", 
            normalization=True,
            test_split_ratio=0.2):


    df = pd.read_csv(data_path)
    # Drop the 0 amount transactions
    df = df.drop(df[df['Amount'] == 0].index)

    if normalization:
        # Normalize each column
        for column in df.columns:
            if column != "Class":
                df[column] = (df[column] - df[column].mean()) / df[column].std()

    # X and y are features and labels
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    X_pos = X[y == 1].values
    y_pos = y[y == 1].values
    X_neg = X[y == 0].values
    y_neg = y[y == 0].values

    # Split into train (80%) and temp (20%)
    X_neg_train, X_neg_temp, y_neg_train, y_neg_temp = train_test_split(
        X_neg, y_neg, test_size=test_split_ratio, random_state=42)
    X_pos_train, X_pos_temp, y_pos_train, y_pos_temp = train_test_split(
        X_pos, y_pos, test_size=test_split_ratio, random_state=42)

    # Split the temp set into validation (10%) and test (10%)
    X_neg_val, X_neg_test, y_neg_val, y_neg_test = train_test_split(
        X_neg_temp, y_neg_temp, test_size=0.5, random_state=42)
    X_pos_val, X_pos_test, y_pos_val, y_pos_test = train_test_split(
        X_pos_temp, y_pos_temp, test_size=0.5, random_state=42)

    # Combine the negative and positive samples back together
    X_train = np.vstack([X_neg_train, X_pos_train])
    y_train = np.hstack([y_neg_train, y_pos_train])
    X_val = np.vstack([X_neg_val, X_pos_val])
    y_val = np.hstack([y_neg_val, y_pos_val])
    X_test = np.vstack([X_neg_test, X_pos_test])
    y_test = np.hstack([y_neg_test, y_pos_test])

    # Shuffle the training, validation, and test sets to mix the classes
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Convert to PyTorch Tensors
    train_features = torch.from_numpy(X_train).to(torch.float32)
    train_labels = torch.from_numpy(y_train).to(torch.float32)
    val_features = torch.from_numpy(X_val).to(torch.float32)
    val_labels = torch.from_numpy(y_val).to(torch.float32)
    test_features = torch.from_numpy(X_test).to(torch.float32)
    test_labels = torch.from_numpy(y_test).to(torch.float32)

    return train_features, train_labels, \
                val_features, val_labels, \
                    test_features, test_labels

# provides a function that outputs your dataloaders here: train, val, test,
def make_dataset(data_path, 
                 normalization, 
                 test_split_ratio):

    train_features, train_labels, \
                val_features, val_labels, \
                    test_features, test_labels = process(data_path, normalization, test_split_ratio)

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    return train_dataset, val_dataset, test_dataset

def make_dataloader(batch_size, ds, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)        