import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import glob,os,sys
from tqdm.auto import tqdm
import proplot as plot
import json,pickle
import dask.array as da
import gc
from sklearn.decomposition import PCA
from tools import derive_var,read_and_proc,preproc_noensemble
from tools.mlr import mlr
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
from tqdm.auto import tqdm
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import vae3d
from vae3d import VAEEncoder, VAEDecoder, VAE, elbo_loss
import optuna
import torch.nn.functional as F


def total_variation(x):
    """
    Compute isotropic total variation for 3D input.
    x: shape (B, C, D, H, W)
    """
    tv_d = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).sum()
    tv_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).sum()
    return tv_d + tv_h + tv_w

def create_high_freq_mask(shape, cutoff_ratio=0.2):
    """
    Create a binary 3D mask for high-frequency components in Fourier space.

    Args:
        shape (tuple): Shape of the 3D volume (D, H, W).
        cutoff_ratio (float): Fraction of maximum radius to start penalizing (e.g., 0.2 = keep low 20% frequencies).

    Returns:
        torch.Tensor: A mask with same shape as input FFT magnitude, with 1s at high frequencies and 0s elsewhere.
    """
    D, H, W = shape
    zz, yy, xx = torch.meshgrid(
        torch.fft.fftfreq(D), 
        torch.fft.fftfreq(H), 
        torch.fft.fftfreq(W), 
        indexing='ij'
    )

    # Compute radius from center frequency
    radius = torch.sqrt(xx**2 + yy**2 + zz**2)

    # Normalize radius to [0, 0.5], then mask above cutoff_ratio
    mask = (radius >= cutoff_ratio).float()

    return mask
    
def fourier_high_freq_penalty(x, cutoff=0.2):
    # x shape: (1, C, D, H, W)
    x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-3, -2, -1))
    freqs = torch.fft.fftfreq(x.shape[-1])  # you can build a radial mask
    high_freq_mask = create_high_freq_mask(x.shape[-3:], cutoff).to(x.device)
    return torch.abs(x_fft_shifted * high_freq_mask).mean()
    
##############################################################################################################
# Method1: Direct optimizating 3D Volume
##############################################################################################################
def find_max_output_from_mean(
    vae_model,
    mean_input,
    num_iterations=150,
    initial_lr=1e-2,
    kl_weight=1.0,
    l2_weight=1e-2,
    tv_weight=0.1,
    stat_weight=10.0,             # NEW: weight for mean/std deviation penalty
    trust_radius=0.5,             # NEW: max L2 norm from mean_input
    device='cuda',
    input_bounds=(0.0, 1.0),
    early_stop_patience=10,
    lr_drop_iter=50,
    lr_drop_factor=0.1,
    add_noise_std=0.01,
    high_freq_weight=0.2
):
    """
    Optimize the input (starting from mean) to maximize VAE scalar output.

    Includes:
        - Input clamping to bounds
        - L2 and total variation regularization
        - Mean/std penalty
        - Trust-region constraint from mean_input
        - Early stopping
    """

    vae_model.eval()
    start_input = mean_input.clone().detach().to(device)
    if add_noise_std > 0:
        noise = add_noise_std * torch.randn_like(start_input)
        start_input = (start_input + noise)
        #start_input = F.avg_pool3d(start_input, kernel_size=5, stride=1, padding=2).detach()

    input_tensor = start_input.requires_grad_()
    optimizer = torch.optim.Adam([input_tensor], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_iter, gamma=lr_drop_factor)

    scalar_history = []
    best_scalar = -float("inf")
    patience_counter = 0

    for i in range(num_iterations):
        optimizer.zero_grad()

        scalar_pred, mu, logvar = vae_model(input_tensor)
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1)
        l2_penalty = torch.norm(input_tensor - mean_input.to(device))
        tv_penalty = total_variation(input_tensor)

        mean_penalty = (input_tensor.mean() - start_input.mean())**2
        std_penalty = (input_tensor.std() - start_input.std())**2

        # Full loss
        loss = -scalar_pred.mean() * 2
        loss += kl_weight * kl.mean()
        loss += l2_weight * l2_penalty
        loss += tv_weight * tv_penalty
        loss += stat_weight * (mean_penalty + std_penalty)
        loss += high_freq_weight * fourier_high_freq_penalty(input_tensor)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp input to bounds
        with torch.no_grad():
            input_tensor.clamp_(*input_bounds)

            # NEW: Trust region constraint
            #delta = input_tensor - mean_input
            #norm = delta.norm()
            #if norm > trust_radius:
            #    input_tensor.copy_(mean_input + delta * (trust_radius / norm))

        current_scalar = scalar_pred.item()
        scalar_history.append(current_scalar)

        if i % 10 == 0:
            print(f"Step {i:3d}: scalar = {current_scalar:.4f} | lr = {scheduler.get_last_lr()[0]:.5f}")

        # Early stopping
        if current_scalar > best_scalar + 1e-4:
            best_scalar = current_scalar
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at step {i} (scalar plateaued).")
                break

    return input_tensor.detach(), scalar_history

import optuna
import torch.nn.functional as F

def objective(trial, vae_model, mean_input, device='cuda'):
    # Hyperparameter search space
    initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-1)
    kl_weight = trial.suggest_float("kl_weight", 1e-3, 10.0)
    l2_weight = trial.suggest_float("l2_weight", 1e-2, 1.0)
    tv_weight = trial.suggest_float("tv_weight", 0.3, 1.0)
    stat_weight = trial.suggest_float("stat_weight", 0.0, 50.0)
    lr_drop_iter = trial.suggest_int("lr_drop_iter", 30, 100)
    lr_drop_factor = trial.suggest_float("lr_drop_factor", 0.1, 0.9)
    add_noise_std = trial.suggest_float("add_noise_std", 0.0, 0.05)
    trust_radius = trial.suggest_float("trust_radius", 0.1, 1.0)
    high_freq_weight= trial.suggest_float("high_freq_weight", 0.001, 5.0, log=True)

    # Fixed input bounds (recommended)
    input_bounds = (0.0, 1.0)

    _, scalar_history = find_max_output_from_mean(
        vae_model=vae_model,
        mean_input=mean_input,
        num_iterations=150,
        initial_lr=initial_lr,
        kl_weight=kl_weight,
        l2_weight=l2_weight,
        tv_weight=tv_weight,
        stat_weight=stat_weight,
        trust_radius=trust_radius,
        device=device,
        input_bounds=input_bounds,
        early_stop_patience=10,
        lr_drop_iter=lr_drop_iter,
        lr_drop_factor=lr_drop_factor,
        add_noise_std=add_noise_std,
        high_freq_weight=high_freq_weight
    )

    return max(scalar_history)

def run_optuna_study(vae_model, mean_input, n_trials=50, device='cuda'):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, vae_model, mean_input, device), n_trials=n_trials)

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    return study

##############################################################################################################
# Method2: Optimizing PCA coefficients
##############################################################################################################