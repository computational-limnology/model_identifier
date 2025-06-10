# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:55:05 2025

Refactored for causal emulation using an encoder-decoder with single-parent mapping.
"""


import os 

os.chdir("/Users/au740615/Documents/projects/model_identifier/")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Parameters
# ------------------------------
latent_dim = 5
output_dim = 10  # Number of state variables in the dataset
hidden_dim = 64
model_embedding_dim = 3  # Size of model ID embedding
epochs = 100
batch_size = 32
learning_rate = 1e-3

# ------------------------------
# Dataset Loading and Preprocessing
# ------------------------------
df = pd.read_csv("./data/integrated_all.csv", parse_dates=["datetime"])

features = df.drop(columns=["datetime", "Model"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

label_encoder = LabelEncoder()
model_ids = label_encoder.fit_transform(df["Model"])
model_tensor = torch.tensor(model_ids, dtype=torch.long)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(X_scaled, dtype=torch.float32)

dataset = TensorDataset(X_tensor, model_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_dim = X_tensor.shape[1]
output_dim = Y_tensor.shape[1]
num_models = len(label_encoder.classes_)

# ------------------------------
# Encoder
# ------------------------------
class CausalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_models, embedding_dim):
        super().__init__()
        self.model_embedding = nn.Embedding(num_models, embedding_dim)
        self.lstm = nn.LSTM(input_dim + embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, model_id):
        emb = self.model_embedding(model_id).unsqueeze(1)
        x = x.unsqueeze(1)
        emb_x = torch.cat([x, emb], dim=2)
        _, (h_n, _) = self.lstm(emb_x)
        h = h_n[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

# ------------------------------
# Decoder
# ------------------------------
class CausalDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.single_parents = nn.ModuleList([
            nn.Linear(1, 1) for _ in range(output_dim)
        ])

    def forward(self, z):
        outputs = []
        for i, fc in enumerate(self.single_parents):
            zi = z[:, i % latent_dim].unsqueeze(1)
            outputs.append(fc(zi))
        return torch.cat(outputs, dim=1)

# ------------------------------
# Loss Function
# ------------------------------
def vae_loss(recon_y, y, mu, logvar):
    recon_loss = nn.MSELoss()(recon_y, y)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# ------------------------------
# Training
# ------------------------------
encoder = CausalEncoder(input_dim, hidden_dim, latent_dim, num_models, model_embedding_dim)
decoder = CausalDecoder(latent_dim, output_dim)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for xb, model_id, yb in dataloader:
        optimizer.zero_grad()
        z, mu, logvar = encoder(xb, model_id)
        recon_y = decoder(z)
        loss = vae_loss(recon_y, yb, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ------------------------------
# Causal Probing Output
# ------------------------------
for i, fc in enumerate(decoder.single_parents):
    print(f"Output {i} depends on latent dim {i % latent_dim} with weight: {fc.weight.data.numpy()}")

# ------------------------------
# Latent Space Visualization (PCA + t-SNE)
# ------------------------------
encoder.eval()
with torch.no_grad():
    z_all, _, _ = encoder(X_tensor, model_tensor)
    z_all_np = z_all.numpy()

    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(z_all_np)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_coords = tsne.fit_transform(z_all_np)

    # DataFrame for visualization
    df_vis = pd.DataFrame({
        "PCA1": pca_coords[:, 0],
        "PCA2": pca_coords[:, 1],
        "tSNE1": tsne_coords[:, 0],
        "tSNE2": tsne_coords[:, 1],
        "Model": df["Model"]
    })

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=df_vis, x="PCA1", y="PCA2", hue="Model", ax=axs[0])
    axs[0].set_title("PCA of Latent Representations")

    sns.scatterplot(data=df_vis, x="tSNE1", y="tSNE2", hue="Model", ax=axs[1])
    axs[1].set_title("t-SNE of Latent Representations")

    plt.tight_layout()
    plt.show()
