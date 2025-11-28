# pytorch_vae_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json

#torch_pi = torch.tensor(np.log(2 * np.pi), dtype=torch.float32)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class KLAnnealer:
    def __init__(self, patience=5, step=0.1, max_weight=1.0, min_delta=1e-4):
        self.patience = patience
        self.step = step
        self.max_weight = max_weight
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.weight = 0.0

    def update(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.weight = min(self.weight + self.step, self.max_weight)

        return self.weight
        

def conv_output_size(size, kernel_size, stride, padding):
    return (size + 2 * padding - kernel_size) // stride + 1

    
#class Sampling(nn.Module):
#    def forward(self, mean, log_var):
#        if torch.isnan(log_var).any() or torch.isinf(log_var).any():
#            raise ValueError("🚨 log_var contains NaN or Inf BEFORE clamping")
        
#        log_var = torch.clamp(log_var, min=-10.0, max=2.0)

#        std = torch.exp(0.5 * log_var)

#        if torch.isnan(std).any() or torch.isinf(std).any():
#            raise ValueError("🚨 std is NaN or Inf after exp")

#        eps = torch.randn_like(std)
#        z = mean + eps * std

#        if torch.isnan(z).any():
#            raise ValueError("🚨 z is NaN after sampling")

#        return z

class Sampling(nn.Module):
    def forward(self, mean, log_var, training: bool):
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        if training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

class VAEEncoder(nn.Module):
    def __init__(self, encoder_config, input_shape, device):
        super().__init__()

        self.latent_dim = encoder_config["latent_dim"]
        self.activation = getattr(F, encoder_config["activation"].lower())
        
        self.conv1 = nn.Conv3d(1, encoder_config["conv_1"]["filter_num"], tuple(encoder_config["conv_1"]["kernel_size"]), stride=encoder_config["conv_1"]["stride"], padding=tuple(k//2 for k in encoder_config["conv_1"]["kernel_size"]))
        self.conv2 = nn.Conv3d(encoder_config["conv_1"]["filter_num"], encoder_config["conv_2"]["filter_num"], tuple(encoder_config["conv_2"]["kernel_size"]), stride=encoder_config["conv_2"]["stride"], padding=tuple(k//2 for k in encoder_config["conv_2"]["kernel_size"]))
        self.conv3 = nn.Conv3d(encoder_config["conv_2"]["filter_num"], encoder_config["conv_3"]["filter_num"], tuple(encoder_config["conv_3"]["kernel_size"]), stride=encoder_config["conv_3"]["stride"], padding=tuple(k//2 for k in encoder_config["conv_3"]["kernel_size"]))

        self.conv_mu = nn.Conv3d(encoder_config["conv_3"]["filter_num"], encoder_config["conv_mu"]["filter_num"], tuple(encoder_config["conv_mu"]["kernel_size"]), stride=encoder_config["conv_mu"]["stride"], padding=tuple(k//2 for k in encoder_config["conv_mu"]["kernel_size"]))
        self.conv_logvar = nn.Conv3d(encoder_config["conv_3"]["filter_num"], encoder_config["conv_mu"]["filter_num"], tuple(encoder_config["conv_mu"]["kernel_size"]), stride=encoder_config["conv_mu"]["stride"], padding=tuple(k//2 for k in encoder_config["conv_mu"]["kernel_size"]))
        
        # Static shape computation
        d, h, w = input_shape[2:]  # Skip batch and channel
        for key in ["conv_1", "conv_2", "conv_3", "conv_mu"]:
            cfg = encoder_config[key]
            kernel = cfg["kernel_size"]
            stride = cfg["stride"]
            padding = [k // 2 for k in kernel]
            d = conv_output_size(d, kernel[0], stride, padding[0])
            h = conv_output_size(h, kernel[1], stride, padding[1])
            w = conv_output_size(w, kernel[2], stride, padding[2])

        self.flattened_size = encoder_config["conv_mu"]["filter_num"] * d * h * w
        
        #self.flattened_size = encoder_config["conv_mu"]["filter_num"] * np.prod(dims)
        
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)
        self.sampler = Sampling()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        mu = self.conv_mu(x).flatten(start_dim=1)
        logvar = self.conv_logvar(x).flatten(start_dim=1)
        mu = self.fc_mu(mu)
        logvar = self.fc_logvar(logvar)
        logvar = torch.clamp(logvar, min=-10, max=2)  # <- 🔒 clamp BEFORE passing to Sampling
        #logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.sampler(mu, logvar, self.training)
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, decoder_config):
        super().__init__()
        self.latent_dim = decoder_config["latent_dim"]
        self.fc1 = nn.Linear(self.latent_dim, decoder_config["fc1_dim"])
        self.fc2 = nn.Linear(decoder_config["fc1_dim"], 1)
        self.activation = getattr(F, decoder_config["activation"].lower())
        self.norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z):
        z = self.norm(z)
        x = self.activation(self.fc1(z))
        scalar_output = self.fc2(x)
        return scalar_output

# simple, stable decoder: LN -> Linear -> SiLU -> Dropout -> Linear
class DecMLP(nn.Module):
    def __init__(self, decoder_config):#d, h):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[decoder_config['head_activation'].lower()]
        self.net = nn.Sequential(
            nn.LayerNorm(decoder_config["latent_dim"]),
            nn.Linear(decoder_config["latent_dim"], decoder_config["latent_hidden_dim"]),
            nn.SiLU(),
            nn.Dropout(p=decoder_config["dropout_rate"]),
            nn.Linear(decoder_config["latent_hidden_dim"], 1),
        )
    def forward(self, z): return self.net(z)
        
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        # Clamp for numerical sanity (only affects KL; z uses mu when beta==0)
        #logvar_c = logvar.clamp(min=-10.0, max=2.0)

        #if self.training and self.beta > 0.0:
        #    std = torch.exp(0.5 * logvar_c)
        #    eps = torch.randn_like(std)
        #    z = mu + eps * std
        #else:
        #    z = mu  # deterministic
        scalar_output = self.decoder(z)
        return scalar_output, mu, logvar
        # inside VAE.forward
        #import math
        #z, mu, logvar = self.encoder(x)
        #logvar = logvar.clamp(min=-10.0, max=2.0 * math.log(getattr(self, "std_cap", 0.30)))
        #if self.training and getattr(self, "beta", 0.0) > 0.0:
        #    std = torch.exp(0.5 * logvar)
        #    z = mu + torch.randn_like(std) * std * getattr(self, "noise_scale", 1.0)
        #else:
        #    z = mu
        #scalar_output = self.decoder(z)
        #return scalar_output, mu, logvar

class AnnealingScheduler:
    def __init__(self, vae_model, total_epochs):
        self.vae_model = vae_model
        self.total_epochs = total_epochs

    def update(self, current_epoch):
        new_weight = current_epoch / float(self.total_epochs)
        self.vae_model.kl_weight = new_weight

def elbo_loss(scalar_target, scalar_pred, mu, logvar, kl_weight=1.0):
    if not isinstance(logvar, torch.Tensor):
        logvar = torch.tensor(logvar, dtype=mu.dtype, device=mu.device)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, dtype=logvar.dtype, device=logvar.device)
    recon_loss = F.mse_loss(scalar_pred.view(-1), scalar_target.view(-1), reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1)
    return torch.mean(recon_loss + kl_weight * kl)
#class VAEDecoder(nn.Module):
#    def __init__(self, decoder_config):
#        super().__init__()
#        self.latent_dim = decoder_config["latent_dim"]
#        self.reshape_dims = (decoder_config["latent_reshape"]["dim_1"], decoder_config["latent_reshape"]["dim_2"], decoder_config["latent_reshape"]["dim_3"], decoder_config["latent_reshape"]["dim_4"])

#        self.fc = nn.Linear(self.latent_dim, np.prod(self.reshape_dims))

#        self.deconv1 = nn.ConvTranspose3d(self.reshape_dims[0], decoder_config["conv_t_0"]["filter_num"], tuple(decoder_config["conv_t_0"]["kernel_size"]), stride=decoder_config["conv_t_0"]["stride"], padding=tuple(k//2 for k in decoder_config["conv_t_0"]["kernel_size"]))
#        self.deconv2 = nn.ConvTranspose3d(decoder_config["conv_t_0"]["filter_num"], decoder_config["conv_t_1"]["filter_num"], tuple(decoder_config["conv_t_1"]["kernel_size"]), stride=decoder_config["conv_t_1"]["stride"], padding=tuple(k//2 for k in decoder_config["conv_t_1"]["kernel_size"]))
#        self.deconv3 = nn.ConvTranspose3d(decoder_config["conv_t_1"]["filter_num"], decoder_config["conv_t_2"]["filter_num"], tuple(decoder_config["conv_t_2"]["kernel_size"]), stride=decoder_config["conv_t_2"]["stride"], padding=tuple(k//2 for k in decoder_config["conv_t_2"]["kernel_size"]))

#        self.mu_out = nn.ConvTranspose3d(decoder_config["conv_t_2"]["filter_num"], decoder_config["conv_mu"]["filter_num"], tuple(decoder_config["conv_mu"]["kernel_size"]), stride=decoder_config["conv_mu"]["stride"], padding=tuple(k//2 for k in decoder_config["conv_mu"]["kernel_size"]))
#        self.logvar_out = nn.ConvTranspose3d(decoder_config["conv_t_2"]["filter_num"], decoder_config["conv_log_var"]["filter_num"], tuple(decoder_config["conv_log_var"]["kernel_size"]), stride=decoder_config["conv_log_var"]["stride"], padding=tuple(k//2 for k in decoder_config["conv_log_var"]["kernel_size"]))

#        self.activation = getattr(F, decoder_config["activation"].lower())

#    def forward(self, z):
#        x = self.fc(z).view(-1, *self.reshape_dims)
#        x = self.activation(self.deconv1(x))
#        x = self.activation(self.deconv2(x))
#        x = self.activation(self.deconv3(x))
#        mu = self.mu_out(x)
#        logvar = self.logvar_out(x)
#        return mu, logvar