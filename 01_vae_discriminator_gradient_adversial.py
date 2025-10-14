"""
Revised version of 01_vae_discriminator.py
- Correct WGAN-GP style gradient penalty applied to discriminator with interpolated z between encoder outputs and prior samples.
- Proper adversarial training: encoder receives an adversarial loss that tries to *confuse* the discriminator.
- Train/test split (stratified by model label), proper DataLoaders for train and test.
- Logging of losses and metrics to CSV and printing per-epoch summaries.
- Saves model state_dicts to ./output/
- Uses device (GPU if available) and reproducible seeds.

Notes:
- Adjust hyperparameters at the top as needed.
- This script intentionally keeps the classifier as a multi-class discriminator predicting simulator ID from z.
"""

import os
import random
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

os.chdir("/Users/au740615/Documents/projects/model_identifier/")

# -------------------------
# Config / hyperparameters
# -------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_PATH = "./data/integrated_all.csv"
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LENGTH = 10  # increase sequence length compared to original
BATCH_SIZE = 64
LR_VAE = 1e-3
LR_DISC = 1e-3
EPOCHS = 80
KL_WEIGHT = 1e-4  # adjust for stronger variational effect if desired
ADV_WEIGHT = 0.2 # 1.0  # weight for adversarial encoder loss (encoder tries to maximize disc loss)
GP_WEIGHT = 10.0  # gradient penalty weight (WGAN-GP style)
TEST_SIZE = 0.4 # 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Utility: sequence builder
# -------------------------

def build_sequences(df, feature_cols, label_col, seq_len=SEQ_LENGTH):
    X_list = []
    Y_list = []
    groups = df.groupby(label_col)
    for name, g in groups:
        g = g.sort_values("datetime")
        arr = g[feature_cols].values
        n = len(arr)
        if n < seq_len:
            continue
        for i in range(n - seq_len + 1):
            X_list.append(arr[i : i + seq_len])
            Y_list.append(name)
    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.stack(X_list), np.array(Y_list)

# -------------------------
# Models
# -------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # take last layer's hidden state
        h = h_n[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (batch, latent_dim)
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(1)  # (batch, 1, hidden)
        # repeat across seq_len
        dec_in = h0.repeat(1, self.seq_len, 1)
        out, _ = self.lstm(dec_in)
        out = self.out(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z):
        # returns logits (no softmax)
        return self.net(z)

# -------------------------
# Loss helpers
# -------------------------

def vae_loss_function(recon_x, x, mu, logvar, kl_weight=KL_WEIGHT):
    # recon_x, x: (batch, seq_len, features)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # KL divergence between q(z|x)=N(mu, sigma^2) and p(z)=N(0,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # normalize kl per batch
    kl = kl / x.size(0)
    return recon_loss + kl_weight * kl, recon_loss.detach(), kl.detach()


def gradient_penalty(discriminator, real_z, fake_z, device=DEVICE):
    # real_z: samples from prior (N(0,1)), fake_z: encoded samples
    batch_size = real_z.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_z)
    interpolates = alpha * real_z + (1 - alpha) * fake_z
    interpolates = interpolates.requires_grad_(True)
    logits = discriminator(interpolates)
    # For multi-class logits, reduce to a scalar by taking sum of logits for true class or mean.
    # We'll take the sum of the logits along class dim to get a scalar per sample.
    logits_sum = logits.sum(dim=1)
    grads = torch.autograd.grad(outputs=logits_sum,
                                inputs=interpolates,
                                grad_outputs=torch.ones_like(logits_sum, device=device),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    # penalty term to enforce grad_norm close to 1
    penalty = ((grad_norm - 1.0) ** 2).mean()
    return penalty

# -------------------------
# Training loop function
# -------------------------

def train_one_experiment(df, model_names, wq_names, experiment_name="exp"):
    print(f"Starting experiment {experiment_name} | models: {model_names} | features: {wq_names}")

    # filter and preprocess
    df_sub = df[df['Model'].isin(model_names)].copy()
    if df_sub.empty:
        print("No data after filtering models. Exiting.")
        return

    # encode labels
    le = LabelEncoder()
    df_sub['model_label'] = le.fit_transform(df_sub['Model'])
    n_classes = len(le.classes_)

    # print({le.classes_[i]: per_model_total[i] for i in range(n_classes)})

    # scale features
    scaler = MinMaxScaler()
    df_sub[wq_names] = scaler.fit_transform(df_sub[wq_names])

    # build sequences
    X, Y_raw = build_sequences(df_sub, wq_names, label_col='model_label', seq_len=SEQ_LENGTH)
    if X.shape[0] == 0:
        print("No sequences created (not enough data). Exiting.")
        return

    # map Y_raw to integer labels (they already are ints but ensure dtype)
    Y = Y_raw.astype(int)

    # train/test split stratified by label
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE,
    #                                                    random_state=RANDOM_SEED, stratify=Y)

    # 1) time-based split to create train/test dataframes
    df_train, df_test = time_based_split(df_sub, test_frac=0.35)

    # 2) build sequences with stride for independence (e.g., stride=SEQ_LENGTH for fully non-overlapping)
    X_train, y_train = build_sequences(df_train, wq_names, label_col='model_label', seq_len=SEQ_LENGTH, stride=SEQ_LENGTH)
    X_test, y_test = build_sequences(df_test,  wq_names, label_col='model_label', seq_len=SEQ_LENGTH, stride=SEQ_LENGTH)

    # 3) if some classes lack enough test samples, fill up to N_per_class using overlapping windows in test partition
    N_PER_CLASS = 150
    if any([(y_test==i).sum() < N_PER_CLASS for i in np.unique(y_test)]):
        X_test_extra, Y_test_extra = collect_test_sequences_per_class(df_test, wq_names, SEQ_LENGTH, n_per_class=N_PER_CLASS)
        # concatenate and dedupe if necessary
        X_test = np.concatenate([X_test, X_test_extra], axis=0)
        y_test = np.concatenate([y_test, Y_test_extra], axis=0)


    # create dataloaders
    def to_loader(Xa, ya, shuffle=True):
        tx = torch.tensor(Xa, dtype=torch.float32)
        ty = torch.tensor(ya, dtype=torch.long)
        ds = TensorDataset(tx, ty)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=True)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    test_loader = to_loader(X_test, y_test, shuffle=False)

    input_dim = len(wq_names)
    hidden_dim = max(16, 4 * input_dim)
    latent_dim = max(8, 4 * input_dim)

    encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim).to(DEVICE)
    decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len=SEQ_LENGTH).to(DEVICE)
    discriminator = Discriminator(latent_dim, hidden_dim, n_classes).to(DEVICE)

    params_vae = list(encoder.parameters()) + list(decoder.parameters())
    optimizer_vae = torch.optim.Adam(params_vae, lr=LR_VAE)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=LR_DISC)

    criterion_ce = nn.CrossEntropyLoss()

    # logging setup
    metrics = []

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        decoder.train()
        discriminator.train()

        epoch_rec_loss = 0.0
        epoch_kl = 0.0
        epoch_vae_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_adv_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # ----------------
            # VAE forward
            # ----------------
            mu, logvar = encoder(batch_x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # reparameterization

            recon_x = decoder(z)
            vae_loss, rec_loss_val, kl_val = vae_loss_function(recon_x, batch_x, mu, logvar)

            # ----------------
            # Discriminator update
            # ----------------
            # train discriminator to predict true model labels from z.detach()
            optimizer_disc.zero_grad()
            with torch.no_grad():
                z_det = z.detach()
            logits = discriminator(z_det)
            disc_loss = criterion_ce(logits, batch_y)

            # gradient penalty between prior samples and encoder z
            z_real = torch.randn_like(z_det)
            gp = gradient_penalty(discriminator, z_real, z_det)
            disc_loss_total = disc_loss + GP_WEIGHT * gp
            disc_loss_total.backward()
            optimizer_disc.step()

            # ----------------
            # VAE (encoder) update with adversarial loss
            # ----------------
            optimizer_vae.zero_grad()
            # Encoder should both reconstruct (via VAE loss) AND try to *fool* the discriminator.
            # Compute discriminator logits on z (no detach) and adversarial loss for encoder.
            logits_for_adv = discriminator(z)
            adv_loss = criterion_ce(logits_for_adv, batch_y)
            # encoder wants to maximize discr's loss -> minimize negative adv_loss
            enc_total_loss = vae_loss + ADV_WEIGHT * (-adv_loss)
            enc_total_loss.backward()
            optimizer_vae.step()

            epoch_rec_loss += rec_loss_val.item()
            epoch_kl += kl_val.item()
            epoch_vae_loss += vae_loss.item()
            epoch_disc_loss += disc_loss.item()
            epoch_adv_loss += adv_loss.item()

        # end train loop
        n_batches = len(train_loader)
        avg_rec = epoch_rec_loss / n_batches
        avg_kl = epoch_kl / n_batches
        avg_vae = epoch_vae_loss / n_batches
        avg_disc = epoch_disc_loss / n_batches
        avg_adv = epoch_adv_loss / n_batches

        # Evaluate on test set: reconstruction loss and discriminator accuracy
        encoder.eval()
        decoder.eval()
        discriminator.eval()

        test_rec = 0.0
        test_kl = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)
                mu_t, logvar_t = encoder(bx)
                std_t = torch.exp(0.5 * logvar_t)
                z_t = mu_t  # use mu deterministically for evaluation
                recon_t = decoder(z_t)
                recon_loss = F.mse_loss(recon_t, bx, reduction='mean')
                kl_t = -0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp()) / bx.size(0)
                test_rec += recon_loss.item()
                test_kl += kl_t.item()

                logits_t = discriminator(z_t)
                preds = torch.argmax(logits_t, dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)

        test_rec /= len(test_loader)
        test_kl /= len(test_loader)
        test_acc = correct / total if total > 0 else 0.0

        # --- Compute per-model accuracy on test set ---
        per_model_correct = np.zeros(n_classes)
        per_model_total = np.zeros(n_classes)

        with torch.no_grad():
            all_preds = []
            all_true = []
            for bx, by in test_loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)
                mu_t, logvar_t = encoder(bx)
                z_t = mu_t
                logits_t = discriminator(z_t)
                preds = torch.argmax(logits_t, dim=1)
                for i in range(n_classes):
                    mask = (by == i)
                    per_model_correct[i] += (preds[mask] == by[mask]).sum().item()
                    per_model_total[i] += mask.sum().item()
                    
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(by.cpu().numpy())

        per_model_acc = per_model_correct / np.maximum(per_model_total, 1)
        per_model_acc_dict = {le.classes_[i]: per_model_acc[i] for i in range(n_classes)}

        

        metrics.append({
            'epoch': epoch,
            'train_rec_loss': avg_rec,
            'train_kl': avg_kl,
            'train_vae_loss': avg_vae,
            'train_disc_loss': avg_disc,
            'train_adv_loss': avg_adv,
            'test_rec_loss': test_rec,
            'test_kl': test_kl,
            'test_disc_acc': test_acc,
            **{f"acc_{m}": a for m, a in per_model_acc_dict.items()}
        })

        print(f"Epoch {epoch}/{EPOCHS} | train_rec={avg_rec:.5f} train_kl={avg_kl:.5f} "
              f"train_vae={avg_vae:.5f} train_disc={avg_disc:.5f} train_adv={avg_adv:.5f} | "
              f"test_rec={test_rec:.5f} test_kl={test_kl:.5f} test_acc={test_acc:.4f}")

        # save intermediate model every 10 epochs
        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), OUTPUT_DIR / f"encoder_{experiment_name}_ep{epoch}.pth")
            torch.save(decoder.state_dict(), OUTPUT_DIR / f"decoder_{experiment_name}_ep{epoch}.pth")
            torch.save(discriminator.state_dict(), OUTPUT_DIR / f"discriminator_{experiment_name}_ep{epoch}.pth")

    # end epoch loop
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / f"metrics_{experiment_name}.csv", index=False)
    print(f"Experiment finished. Metrics saved to {OUTPUT_DIR / f'metrics_{experiment_name}.csv'}")

    # final save
    torch.save(encoder.state_dict(), OUTPUT_DIR / f"encoder_{experiment_name}_final.pth")
    torch.save(decoder.state_dict(), OUTPUT_DIR / f"decoder_{experiment_name}_final.pth")
    torch.save(discriminator.state_dict(), OUTPUT_DIR / f"discriminator_{experiment_name}_final.pth")

    return metrics_df

# assume df_sub sorted by 'datetime' and contains 'model_label'
def time_based_split(df_sub, test_frac=0.3):
    dfs = []
    dfs_train = []
    dfs_test = []
    for model_label, g in df_sub.groupby('model_label'):
        g = g.sort_values('datetime')
        n = len(g)
        cut = int(n * (1 - test_frac))
        dfs_train.append(g.iloc[:cut])
        dfs_test.append(g.iloc[cut:])
    df_train = pd.concat(dfs_train).sort_values(['model_label','datetime'])
    df_test = pd.concat(dfs_test).sort_values(['model_label','datetime'])
    return df_train, df_test

def build_sequences(df, feature_cols, label_col, seq_len=SEQ_LENGTH, stride=1):
    X_list, Y_list = [], []
    for name, g in df.groupby(label_col):
        g = g.sort_values("datetime")
        arr = g[feature_cols].values
        n = len(arr)
        for i in range(0, n - seq_len + 1, stride):
            X_list.append(arr[i:i+seq_len])
            Y_list.append(name)
    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.stack(X_list), np.array(Y_list)

def collect_test_sequences_per_class(df_sub, feature_cols, seq_len, n_per_class, min_gap=0):
    # min_gap ensures non-overlap between chosen sequences (in rows)
    Xs, Ys = [], []
    for label, g in df_sub.groupby('model_label'):
        g = g.sort_values('datetime').reset_index(drop=True)
        arr = g[feature_cols].values
        n = len(arr)
        starts = list(range(0, n - seq_len + 1, seq_len + min_gap))  # non-overlapping starts
        # if not enough, allow overlapping by stepping 1
        if len(starts) < n_per_class:
            starts = list(range(0, n - seq_len + 1, 1))
        # randomly select up to n_per_class starts without replacement
        import random
        random.shuffle(starts)
        chosen = starts[:min(n_per_class, len(starts))]
        for s in chosen:
            Xs.append(arr[s:s+seq_len])
            Ys.append(label)
    return np.stack(Xs), np.array(Ys)


# -------------------------
# Main driver
# -------------------------
if __name__ == "__main__":
    # load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Update DATA_PATH variable.")

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])  # expects 'Model' column
    df = df.sort_values(["Model", "datetime"])  # ensure ordering

    # example scenarios (adapt to your original script's scenarios)
    model_scenarios = [
        ["GLM", "WET", "SELMAPROTBAS"],
        ["GLM", "WET"],
        ["SELMAPROTBAS", "WET"],
        ["GLM", "SELMAPROTBAS"],
    ]

    aem_scenarios = {
        "T": ["temp"],
        "DO": ["o2"],
        "T-N": ["temp", "no3","nh4"],
        "T-P": ["temp", "po4"],
        "T+NUTR": ["temp", "no3", "po4"],
        "T+NUTR+MIN": ["temp", "no3", "po4", "nh4", "si"],
        "T+CHL": ["temp", "no3", "po4", "nh4", "si", "o2", "chla", "zoop"],
    }

    # run experiments
    for mi, model_list in enumerate(model_scenarios):
        for name, features in aem_scenarios.items():
            exp_name = f"mi{mi}_{name}"
            try:
                
                train_one_experiment(df, model_list, features, experiment_name=exp_name)
            except Exception as e:
                print(f"Experiment {exp_name} failed: {e}")

    print("All experiments complete.")

    
