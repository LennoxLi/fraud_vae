"""Handles Re-sampling of minority class"""
import torch
import numpy as np

from imblearn.over_sampling import SMOTE
from collections import Counter

# implement a function that modifies original train dataset and outputs the augmented trainloader
from torch.utils.data import TensorDataset

vae_model_path = "./outputs/vae_exp_fraud_vae.pt"

def augment_trainset(trainset, method):

    X, y = trainset.tensors[0].numpy(), trainset.tensors[1].numpy()
    print('original trainset distribution %s' % Counter(y))

    if method == "smote":
        sm = SMOTE(random_state=42)
        a_X, a_y = sm.fit_resample(X, y)
        print('after applying SMOTE: %s' % Counter(a_y))
        a_trainset = TensorDataset(torch.from_numpy(a_X), torch.from_numpy(a_y))
        return a_trainset

    elif method == "vae":
        from fraud_vae.models import VAE
        fraud_vae = VAE(input_size=29,
                    hidden_dim=128,
                    latent_size=15)
        fraud_vae.load_state_dict(torch.load(vae_model_path, map_location='cpu'))
        fraud_vae.eval()

        num_synthesis_samples = int(len(y) - np.sum(y) - np.sum(y)) # number of fraud examples to generate to balance the dataset
        X_pos_vae_sampled = fraud_vae.generate(num_synthesis_samples)
        assert X_pos_vae_sampled.shape[0] == num_synthesis_samples
        print(f"sampled {X_pos_vae_sampled.shape[0]} # of fraud examples using VAE")

        a_X = np.concatenate((X, X_pos_vae_sampled.numpy()), axis=0)
        a_y = np.concatenate((y, np.ones(num_synthesis_samples)), axis=0)
        print('after applying VAE: %s' % Counter(a_y))
        a_trainset = TensorDataset(torch.from_numpy(a_X), torch.from_numpy(a_y))
        return a_trainset

    else:
        raise ValueError("Invalid method")