# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:55:05 2025
Modified: 2025-11-13
@author: au740615 (modified)
"""
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

os.chdir("/Users/au740615/Documents/projects/model_identifier")

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Experiment settings ----------
file_path = "./data/integrated_all.csv"

model_scenario_names =  ["All", "GLM-WET", "GLM-SELMA", "WET-SELMA"]
aem_scenario_names = ["ZOOP","CHLA","NO3", "PO4", "O2","NH4","RSI","T", "T+NUTR",
                      "T+NUTR+MINOR", "T+NUTR+MINOR+SYSTEM"]

df_accuracy = pd.DataFrame(columns=['model_scenario',
                                    'aem_scenario', 'model',
                                    'accuracy',
                                    'num'])

max_iter = 10  # number of repeated runs (kept but models are re-initialized per run intentionally)
num_epochs = 50
batch_size = 64
seq_length = 3  # you might want to increase this for real data (e.g., 10, 30)
kl_weight = 1e-3  # increased from 1e-5
lambda_gp = 10.0  # gradient penalty weight for discriminator
lambda_cls_for_encoder = 1.0  # classification weight added to encoder loss

# --------- Helper: network class definitions (outside loops) ----------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # last layer hidden state
        mu = self.fc_mu(h_n)
        logvar = torch.clamp(self.fc_logvar(h_n), min=-10, max=10)
        return mu, logvar


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMDecoder, self).__init__()
        # We'll feed repeated latent vectors into an LSTM to decode sequence
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        # z: (batch, latent_dim)
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)
        lstm_out, _ = self.lstm(z_rep)
        return self.fc_out(lstm_out)  # (batch, seq_len, output_dim)


class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_models):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_models)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)  # raw logits for CrossEntropyLoss


# ---------- Main loops ----------
for model_scenario in model_scenario_names:
    for aem_scenario in aem_scenario_names:

        # load data per scenario iteration
        df = pd.read_csv(file_path, parse_dates=["datetime"])

        # Sort data by Model & datetime
        df = df.sort_values(by=["Model", "datetime"]).reset_index(drop=True)

        # select model names according to scenario
        if model_scenario == "All":
            model_names = ['GLM', 'SELMAPROTBAS', 'WET']
        elif model_scenario == "GLM-WET":
            model_names = ['GLM', 'WET']
        elif model_scenario == "GLM-SELMA":
            model_names = ['GLM', 'SELMAPROTBAS']
        elif model_scenario == "WET-SELMA":
            model_names = ['SELMAPROTBAS', 'WET']
        else:
            model_names = df["Model"].unique().tolist()

        df = df[df['Model'].isin(model_names)].copy()

        # select wq variables for the aem scenario
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
        else:
            # fallback - use all numeric water quality columns if not matched
            wq_names = [c for c in df.columns if c.lower() in ['temp','no3','po4','nh4','si','o2','chla','zoop']]

        if len(wq_names) == 0:
            print(f"No water quality vars selected for scenario {aem_scenario}, skipping.")
            continue

        # Encode model labels
        label_encoder = LabelEncoder()
        df["Model_int"] = label_encoder.fit_transform(df["Model"].astype(str))
        model_classes = label_encoder.classes_
        num_models = len(model_classes)

        # Normalize WQ variables (fit per scenario)
        scaler = MinMaxScaler()
        # handle possible NaNs: drop rows where selected wq are NaN
        df = df.dropna(subset=wq_names + ["datetime", "Model"])
        df[wq_names] = scaler.fit_transform(df[wq_names])

        # numeric time if needed later
        df["time_numeric"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()

        # Prepare sequences per model (prevent mixing across model boundaries)
        X_sequences = []
        Y_labels = []

        for model_id in df["Model_int"].unique():
            model_df = df[df["Model_int"] == model_id].sort_values("datetime")
            vals = model_df[wq_names].values  # shape (n_time, n_features)
            # rolling windows
            for i in range(len(vals) - seq_length):
                seq = vals[i : i + seq_length]
                X_sequences.append(seq)
                Y_labels.append(model_id)

        X_sequences = np.array(X_sequences, dtype=np.float32)
        Y_labels = np.array(Y_labels, dtype=np.int64)

        if len(X_sequences) == 0:
            print(f"No sequences generated for scenario {model_scenario} / {aem_scenario}.")
            continue

        print(f"Scenario {model_scenario} / {aem_scenario} -> X shape: {X_sequences.shape}, labels: {np.unique(Y_labels)}")

        # Convert to tensors and dataset
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_labels, dtype=torch.long)
        dataset = TensorDataset(X_tensor, Y_tensor)

        # train/test split
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        test_len = total_len - train_len
        train_dataset, test_dataset = random_split(dataset, [train_len, test_len],
                                                   generator=torch.Generator().manual_seed(SEED))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = len(wq_names)
        hidden_dim = max(16, 4 * input_dim)
        latent_dim = max(8, 4 * input_dim)
        output_dim = input_dim

        # run repeated experiments (re-initialize models each repetition)
        for num_iter in range(max_iter):
            # instantiate models
            encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim).to(device)
            decoder = LSTMDecoder(latent_dim, hidden_dim, output_dim).to(device)
            discriminator = Discriminator(latent_dim, num_models).to(device)

            # optimizers
            vae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
            disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

            mse_loss = nn.MSELoss()
            cross_entropy = nn.CrossEntropyLoss()

            # training loop
            for epoch in range(num_epochs):
                encoder.train()
                decoder.train()
                discriminator.train()

                epoch_vae_loss = 0.0
                epoch_disc_loss = 0.0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)  # (batch, seq_len, input_dim)
                    batch_y = batch_y.to(device)

                    # ------------------
                    # Forward through encoder (sampling)
                    # ------------------
                    mu, logvar = encoder(batch_x)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + std * eps  # (batch, latent_dim)

                    # Reconstruction
                    recon_x = decoder(z, batch_x.shape[1])  # (batch, seq_len, output_dim)
                    recon_loss = mse_loss(recon_x, batch_x)

                    # KL
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    vae_loss = recon_loss + kl_weight * kl_loss

                    # ------------------
                    # Update discriminator (use detached z but allow GP -> requires_grad on z_det)
                    # ------------------
                    z_det = z.detach().clone().requires_grad_(True)
                    pred_logits = discriminator(z_det)  # logits
                    disc_ce = cross_entropy(pred_logits, batch_y)

                    # gradient penalty: grad of discriminator outputs w.r.t. z_det
                    grads = autograd.grad(outputs=pred_logits.sum(), inputs=z_det, create_graph=True)[0]
                    grad_norm = grads.norm(2, dim=1)
                    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()

                    disc_loss_total = disc_ce + gp

                    disc_optimizer.zero_grad()
                    disc_loss_total.backward()
                    disc_optimizer.step()

                    # ------------------
                    # Update encoder+decoder (include classification loss as additional objective)
                    # Freeze discriminator weights for this step (so we don't update them when backprop class loss)
                    # ------------------
                    for p in discriminator.parameters():
                        p.requires_grad = False

                    pred_logits_for_enc = discriminator(z)  # discriminator run in eval mode wrt gradients
                    class_loss_for_enc = cross_entropy(pred_logits_for_enc, batch_y)

                    total_encoder_loss = vae_loss + lambda_cls_for_encoder * class_loss_for_enc

                    vae_optimizer.zero_grad()
                    total_encoder_loss.backward()
                    vae_optimizer.step()

                    # unfreeze discriminator
                    for p in discriminator.parameters():
                        p.requires_grad = True

                    epoch_vae_loss += vae_loss.item() * batch_x.size(0)
                    epoch_disc_loss += disc_loss_total.item() * batch_x.size(0)

                epoch_vae_loss /= train_len
                epoch_disc_loss /= train_len
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"[{model_scenario} | {aem_scenario}] Iter {num_iter+1}/{max_iter} Epoch {epoch+1}/{num_epochs} - VAE loss: {epoch_vae_loss:.6f}, Disc loss: {epoch_disc_loss:.6f}")

            print("Training complete for this repetition.")

            # ---------- Evaluate discriminator on test set ----------
            encoder.eval()
            discriminator.eval()
            correct = 0
            total = 0
            per_model_correct = defaultdict(int)
            per_model_total = defaultdict(int)

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    mu, logvar = encoder(batch_x)
                    z_test = mu  # use mean for deterministic representation at eval
                    logits = discriminator(z_test)
                    preds = logits.argmax(dim=1)
                    correct_batch = (preds == batch_y).sum().item()
                    correct += correct_batch
                    total += batch_y.size(0)
                    for t, p in zip(batch_y.cpu().numpy(), preds.cpu().numpy()):
                        per_model_total[t] += 1
                        if t == p:
                            per_model_correct[t] += 1

            accuracy = correct / total if total > 0 else 0.0
            print(f"Test Discriminator Accuracy (iter {num_iter}): {accuracy:.4%}")

            # save per-model accuracies into df_accuracy
            for model_id in sorted(per_model_total.keys()):
                model_name = label_encoder.inverse_transform([model_id])[0]
                acc = per_model_correct[model_id] / per_model_total[model_id] if per_model_total[model_id] > 0 else np.nan
                df_accuracy = pd.concat([df_accuracy, pd.DataFrame({
                    'model_scenario': [model_scenario],
                    'aem_scenario': [aem_scenario],
                    'model': [model_name],
                    'accuracy': [acc],
                    'num': [num_iter]
                })], ignore_index=True)

        # end of repetitions
        # save intermediate results
        os.makedirs("./output", exist_ok=True)
        df_accuracy.to_csv("./output/accuracy.csv", index=False)
        print(f"Saved accuracy.csv (scenario {model_scenario}/{aem_scenario}).\n")

print("All done. Final accuracy dataframe saved to ./output/accuracy.csv")
