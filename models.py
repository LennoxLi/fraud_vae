"""Main Models"""
import torch
import torch.nn as nn
# predictive models: mlp, logistic regression, xgboost
# models handling imbalance: vae



# class MLP
class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=512,
                 num_layers=4,
                 dropout_rate=.1):
        super(MLP, self).__init__()

        blocks = []
        for i in range(num_layers):
            in_size, out_size = input_size if i==0 else hidden_size//2**(i-1), hidden_size//2**i
            blocks.extend(
                [
                    nn.Linear(in_size, out_size),
                    nn.LayerNorm(out_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        blocks.extend([
            nn.Linear(hidden_size//2**(num_layers-1), 1),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


# class VAE

class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_size=15, num_layers=2, use_sigmoid=False):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.use_sigmoid = use_sigmoid

        # Build the encoder
        encoder_layers = [nn.Linear(self.input_size, self.hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        # Build the decoder
        decoder_layers = [nn.Linear(self.latent_size, self.hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            decoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(nn.Linear(self.hidden_dim, self.input_size))
        if self.use_sigmoid:
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x_hidden = self.encoder(x)
        mu, logvar = self.mu_layer(x_hidden), self.logvar_layer(x_hidden)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn((num_samples, self.latent_size), device=self.mu_layer.weight.device)
            generated_data = self.decoder(z)
        return generated_data

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def vae_loss(x_hat, x, mu, logvar):
    RE = torch.nn.functional.mse_loss(x_hat, x, reduction="sum")
    KL = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = RE + KL
    return loss, RE, KL
