import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from fraud_vae.models import VAE, vae_loss
from fraud_vae.data import process, make_dataloader
from torch.utils.data import TensorDataset

import os
import argparse

def train(
        exp_name = "vae_experiment",
        data_path = "./data/creditcard.csv",
        num_epochs = 2500,
        latent_size = 15,
        hidden_size = 128,
        learning_rate = 1e-3,
        batch_size = 64,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Process the dataset and retrieve the features and labels
    train_features, train_labels, _, _, _, _ = process(data_path=data_path, normalization=True, test_split_ratio=0.2)

    # Select only the positive examples (fraudulent data) for VAE training
    X_train_pos = train_features[train_labels == 1]
    y_train_pos = train_labels[train_labels == 1]

    print("Number of positive examples in training set: ", X_train_pos.shape)

    # Create the TensorDataset using the positive examples
    train_dataset = TensorDataset(X_train_pos, y_train_pos)

    # Create the DataLoader using the make_dataloader function
    vae_train_loader = make_dataloader(batch_size=batch_size, ds=train_dataset, shuffle=True)

    # Initialize the VAE model
    fraud_vae = VAE(input_size=X_train_pos.shape[1],
                    hidden_dim=hidden_size,
                    latent_size=latent_size).to(device)
    
    # Set the model to training mode
    fraud_vae.train()
    # Optimizer
    optimizer = optim.Adam(fraud_vae.parameters(), lr=learning_rate)
    # Training history
    loss_hist, recon_loss_hist, kl_loss_hist = [], [], []
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        for batch_idx, (data, label) in enumerate(vae_train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            x_hat, mu, logvar = fraud_vae(data)

            # Compute loss
            loss, RE, KL = vae_loss(x_hat, data, mu, logvar)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += RE.item()
            train_kl_loss += KL.item()

        # Logging and monitoring
        if epoch % 100 == 0:
            print(f'Train Epoch: {epoch} \tLoss: {train_loss/len(vae_train_loader):.4f} \tRE: {train_recon_loss/len(vae_train_loader):.4f} \tKL: {train_kl_loss/len(vae_train_loader):.4f}')

        # Save loss history
        loss_hist.append(train_loss / len(vae_train_loader))
        recon_loss_hist.append(train_recon_loss / len(vae_train_loader))
        kl_loss_hist.append(train_kl_loss / len(vae_train_loader))

    # Plot the training losses
    plt.figure(figsize=(12, 6))
    plt.plot(loss_hist, label="Total Loss [RE + KL]")
    plt.plot(recon_loss_hist, label="Reconstruction Loss [RE]")
    plt.plot(kl_loss_hist, label="KL Divergence [KL]")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    if not os.path.exists("./outputs"): os.makedirs("./outputs")
    plt.savefig(f"./outputs/{exp_name}_loss.png", dpi=200)

    # Save the trained model
    torch.save(fraud_vae.state_dict(), "./outputs/{exp_name}_fraud_vae.pt")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="train_vae_fraud", type=str)
    parser.add_argument("--num_epochs", default=2500, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--latent_size", default=15, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    train(
        exp_name=args.exp_name, 
        num_epochs=args.num_epochs,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    

"""
python train_vae.py \
    --exp_name vae_exp \
    --num_epochs 2500 \
    --lr 1e-3 \
    --batch_size 32 \
    --hidden_size 128 \
    --latent_size 15

"""