# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:55:05 2025

@author: au740615
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load the dataset
file_path = "./data/integrated_all.csv"

model_scenario_names =  ["All"]#, "GLM-WET", "GLM-SELMA", "WET-SELMA"]
aem_scenario_names = ["ZOOP","CHLA","NO3", "PO4", "O2","NH4","RSI","T", "T+NUTR", 
                "T+NUTR+MINOR", "T+NUTR+MINOR+SYSTEM"]
df_accuracy = pd.DataFrame(columns=['model_scenario',
                                    'aem_scenario', 'model',
                                    'accuracy',
                                    'num'])
max_iter = 1

for model_scenario in model_scenario_names:
    
    for aem_scenario in aem_scenario_names:
        # load in data
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        
        # Sort data by time for each model
        df = df.sort_values(by=["Model", "datetime"]).reset_index(drop=True)
        
        # check model names
        if model_scenario == "All":
            model_names = ['GLM', 'SELMAPROTBAS', 'WET']
        elif model_scenario == "GLM-WET":
            model_names = ['GLM', 'WET']
        elif model_scenario == "GLM-SELMA":
            model_names = ['GLM', 'SELMAPROTBAS']
        elif model_scenario == "WET-SELMA":
            model_names = ['SELMAPROTBAS', 'WET']
        
        
        df = df[df['Model'].isin(model_names)]
        
        # water quality variables of interest
        if aem_scenario == "T":
            wq_names = ["temp"]
        elif aem_scenario == "T+NUTR":
            wq_names = ["temp", "no3", "po4"]
        elif aem_scenario == "T+NUTR+MINOR":
            wq_names = ["temp", "no3", "po4", "nh4", "si"]
        elif aem_scenario == "T+NUTR+MINOR+SYSTEM":
            wq_names = ["temp", "no3", "po4", "nh4", "si", "o2", "chla", "zoop"]
        elif aem_scenario == "NO3":
            wq_names = ["no3"]
        elif aem_scenario == "PO4":
            wq_names = ["po4"]
        elif aem_scenario == "NH4":
            wq_names = ["nh4"]
        elif aem_scenario == "O2":
            wq_names = ["o2"]
        elif aem_scenario == "RSI":
            wq_names = ["si"]
        elif aem_scenario == "CHLA":
            wq_names = ["chla"]
        elif aem_scenario == "ZOOP":
            wq_names = ["zoop"]
        
        # Encode model labels
        label_encoder = LabelEncoder()
        df["Model"] = label_encoder.fit_transform(df["Model"])  # Convert model names to integers
        
        # Normalize water quality variables
        scaler = MinMaxScaler()
        df[wq_names] = scaler.fit_transform(df[wq_names])
        
        # Convert datetime to numerical values (e.g., seconds since first record)
        df["time_numeric"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()
        
        # Define sequence length (e.g., 10 days of data per sequence)
        seq_length = 3 #10
        
        # Prepare sequences for LSTM
        X_sequences = []
        Y_labels = []
        
        # Process each model separately to avoid mixing sequences across models
        for model_id in df["Model"].unique():
            model_df = df[df["Model"] == model_id][wq_names].values  # Use only target features
            
            for i in range(len(model_df) - seq_length):
                X_sequences.append(model_df[i : i + seq_length])  # Extract rolling window
                Y_labels.append(model_id)  # Store corresponding model label
        
        # Convert to numpy arrays
        X_sequences = np.array(X_sequences)  # Shape: (num_samples, seq_length, 3)
        Y_labels = np.array(Y_labels)  # Shape: (num_samples,)
        
        Counter(Y_labels).values() 
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_labels, dtype=torch.long)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        model_mapping = {model: idx for idx, model in enumerate(df["Model"].unique())}
        print(f"Shape of X: {X_tensor.shape}, Shape of Y: {Y_tensor.shape}")
        
        for num_iter in range(max_iter):
            
            

            
            # Define LSTM-based Encoder
            class LSTMEncoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
                    super(LSTMEncoder, self).__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                    self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                    self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
                
                def forward(self, x):
                    _, (h_n, _) = self.lstm(x)
                    h_n = h_n[-1]
                    mu = self.fc_mu(h_n)
                    logvar = torch.clamp(self.fc_logvar(h_n), min=-10, max=10)
                    return mu, logvar
            
            # Define LSTM-based Decoder
            class LSTMDecoder(nn.Module):
                def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=1):
                    super(LSTMDecoder, self).__init__()
                    self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
                    self.fc_out = nn.Linear(hidden_dim, len(wq_names))  # Output 3 values: (o2, no3, po4)
               
                def forward(self, z, seq_len):
                    z = z.unsqueeze(1).repeat(1, seq_len, 1)  # Shape (batch, seq_len, latent_dim)
                    lstm_out, _ = self.lstm(z)
                    return self.fc_out(lstm_out)
            
            # Define the Discriminator
            class Discriminator(nn.Module):
                def __init__(self, latent_dim, num_models):
                    super(Discriminator, self).__init__()
                    self.fc1 = nn.Linear(latent_dim, 64)  # Increase neurons
                    self.fc2 = nn.Linear(64, 32)  # Add another hidden layer
                    self.fc3 = nn.Linear(32, num_models)
                    self.relu = nn.ReLU()
                    self.softmax = nn.Softmax(dim=1)
                    
                def forward(self, z):
                    h = self.relu(self.fc1(z))
                    h = self.relu(self.fc2(h))
                    return self.softmax(self.fc3(h))
            
            # Set dimensions
            input_dim = len(wq_names)  # (o2, no3, po4)
            hidden_dim = 16
            latent_dim = 4 * 4
            output_dim = len(wq_names)  # Predicting o2, no3, po4
            num_models = len(model_mapping)
            
            # Initialize models
            encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
            decoder = LSTMDecoder(latent_dim, hidden_dim, output_dim)
            discriminator = Discriminator(latent_dim, num_models)
            
            # Define optimizers
            vae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
            disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
            
            # Define loss functions
            mse_loss = nn.MSELoss()
            cross_entropy = nn.CrossEntropyLoss()
            
            def vae_loss_function(recon_x, x, mu, logvar):
                kl_weight = 1E-5 #1E-6
                recon_loss = mse_loss(recon_x, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                return recon_loss + kl_weight * kl_loss
            
            # Training Loop
            num_epochs = 50
            for epoch in range(num_epochs):
                for batch_x, batch_y in dataloader:
                    mu, logvar = encoder(batch_x)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    # z = mu + std * torch.randn_like(std) + (batch_y.float().unsqueeze(1) * 0.5)  # Small shift based on model ID

            
                    recon_x = decoder(z, batch_x.shape[1])
            
                    vae_loss = vae_loss_function(recon_x, batch_x, mu, logvar)
                    pred_y = discriminator(z.detach())
                    disc_loss = cross_entropy(pred_y, batch_y)
                    
                    lambda_gp = 10  # Adjust based on performance
                    gradient_penalty = lambda_gp * (torch.norm(torch.autograd.grad(
                        disc_loss, pred_y, retain_graph=True)[0]) - 1) ** 2
                    disc_loss += gradient_penalty
                    
                    vae_optimizer.zero_grad()
                    vae_loss.backward()
                    vae_optimizer.step()
                    
                    disc_optimizer.zero_grad()
                    disc_loss.backward()
                    disc_optimizer.step()
                
                print(f"Epoch [{epoch+1}/{num_epochs}] - VAE Loss: {vae_loss.item():.4f} - Discriminator Loss: {disc_loss.item():.4f}")
            
            print("Training Complete!")
            
            # Extract latent space representations
            latent_vectors = []
            labels = []
            with torch.no_grad():
                mu_vals, _ = encoder(X_tensor)
                print("Latent Space Mean:", mu_vals.mean(dim=0))
                print("Latent Space Variance:", mu_vals.var(dim=0))  # Should not be too small
                for batch_x, batch_y in dataloader:
                    mu, _ = encoder(batch_x)
                    latent_vectors.append(mu.numpy())
                    labels.append(batch_y.numpy())
            
            latent_vectors = np.vstack(latent_vectors)
            labels = np.hstack(labels)
            
            # Visualizations, take a long time to compute, therefore they are commented out for the loop:
            # PCA Visualization
            # pca = PCA(n_components=2)
            # pca_latent = pca.fit_transform(latent_vectors)
            # plt.figure(figsize=(8, 6))
            # for i, label in enumerate(np.unique(labels)):
            #     plt.scatter(pca_latent[labels == label, 0], pca_latent[labels == label, 1], label=f"Model {label}")
            # plt.legend()
            # plt.title("PCA of Latent Space")
            # plt.show()
            
            # # t-SNE Visualization
            # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            # tsne_latent = tsne.fit_transform(latent_vectors)
            # plt.figure(figsize=(8, 6))
            # for i, label in enumerate(np.unique(labels)):
            #     plt.scatter(tsne_latent[labels == label, 0], tsne_latent[labels == label, 1], label=f"Model {label}")
            # plt.legend()
            # plt.title("t-SNE of Latent Space")
            # plt.show()
            
            # Compute Discriminator Accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    mu, _ = encoder(batch_x)
                    pred_y = discriminator(mu).argmax(dim=1)
                    correct += (pred_y == batch_y).sum().item()
                    total += batch_y.size(0)
            
            accuracy = correct / total
            print(f"Discriminator Accuracy: {accuracy:.2%}")
            
            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    mu, _ = encoder(batch_x)
                    pred_y = discriminator(mu).argmax(dim=1)
                    print(f"Predictions: {pred_y[:10]} | True Labels: {batch_y[:10]}")

            # Compute per-model accuracy
            from collections import defaultdict
            
            correct_counts = defaultdict(int)
            total_counts = defaultdict(int)
            
            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    mu, _ = encoder(batch_x)
                    pred_y = discriminator(mu).argmax(dim=1)
                    
                    for true_label, pred_label in zip(batch_y.numpy(), pred_y.numpy()):
                        total_counts[true_label] += 1
                        if true_label == pred_label:
                            correct_counts[true_label] += 1
            
            # Print per-model accuracy
            print("\nPer-Model Discriminator Accuracy:")
            for model_id in sorted(total_counts.keys()):
                accuracy = correct_counts[model_id] / total_counts[model_id]
                model_name = label_encoder.inverse_transform([model_id])[0]  # Convert ID back to original model name
                print(f"Model {model_name}: {accuracy:.2%}")
                acc = pd.DataFrame({'model_scenario': model_scenario,
                             'aem_scenario': aem_scenario, 
                             'model': model_name,
                             'accuracy': [accuracy],
                             'num': num_iter})
                df_accuracy = pd.concat([df_accuracy, acc], ignore_index = True)
                
            
            
    print(df_accuracy)
    df_accuracy.to_csv("./output/accuracy.csv", index = False)
