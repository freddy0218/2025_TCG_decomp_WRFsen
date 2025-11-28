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
import random
import os
from scipy.special import jn_zeros, j0, j1, jn # jn for m > 0 Bessel functions
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def setup_folds(n_numbers=20,indepedent_test={10,17},val_size=4,seed=42):
    # --- Setup ---
    n_members = n_numbers
    independent_test = indepedent_test   # excluded members
    val_size = val_size
    random.seed(seed)               # for reproducibility; remove or change for new splits
    # --- Eligible members for cross-validation ---
    members = [m for m in range(0, n_members) if m not in independent_test]
    random.shuffle(members)
    n_folds = int(np.ceil(len(members) / val_size))
    # --- Generate folds ---
    folds = []
    for i in range(n_folds):
        start = i * val_size
        end = start + val_size
        val_members = members[start:end]

        # If last fold has fewer than val_size, wrap around
        if len(val_members) < val_size:
            val_members += members[:val_size - len(val_members)]
        folds.append(val_members)

    # --- Print results ---
    for i, f in enumerate(folds, 1):
        print(f"Fold {i}: validation members = {sorted(f)}")
    return folds
    
def get_adjusted_rads(heating=None,p_full=None,dr=3000.0,dtheta_deg=1.0,verbose=False):
    TEST = heating.reshape(heating.shape[0],10,360,208)
    data,offsets = [],[]
    for i in range(TEST.shape[0]):
        temp1,temp2 = adjust_rads_polar_uniform(heating=TEST[i,...], p_full=p_full[i,...], dr=dr, dtheta_deg=dtheta_deg, verbose=verbose)
        data.append(temp1)
        offsets.append(temp2)
    return np.asarray(data), np.asarray(offsets)
    
def adjust_rads_polar_uniform(heating=None, p_full=None, dr=3000.0, dtheta_deg=1.0, cp=1004.0, g=9.81, verbose=True):
    """
    Adjust radiative heating by a global offset so that the *volume-integrated*
    heating (in polar coordinates) is zero, assuming uniform grid spacing.

    Parameters
    ----------
    heating : ndarray
        Radiative heating [K/s or W/kg], shape (n_p, n_r, n_theta)
    p_full : ndarray
        Pressure [Pa], shape (n_p, n_r, n_theta)
    dr : float
        Radial grid spacing [m], default 3000 (3 km)
    dtheta_deg : float
        Angular grid spacing [deg], default 1 degree
    verbose : bool
        If True, print total weighted heating before and after adjustment.

    Returns
    -------
    heating_adjusted : ndarray
        Heating field with volume-integrated mean removed.
    offset : float
        The global offset applied to heating.
    """

    dtheta = np.deg2rad(dtheta_deg)

    n_p, n_r, n_theta = heating.shape
    r = (np.arange(n_r) + 0.5) * dr  # approximate ring centers

    # Pressure thickness (dp) in Pa
    dp = (p_full[:-1, :, :] - p_full[1:, :, :]) * 100.0
    # If heating has same number of p levels as p_full, pad dp
    if dp.shape[0] != heating.shape[0]:
        dp = np.pad(dp, ((0,1),(0,0),(0,0)), mode='edge')

    # Volume weights ∝ r * dr * dθ * dp
    R = r[:, None]  # shape (n_r, 1)
    weight_3d = (cp / g) * dp * (R[None, :, :] * dr * dtheta)

    # Compute total weighted energy before correction
    total_before = np.sum(heating * weight_3d)

    # Global offset so weighted sum = 0
    offset = -total_before / np.sum(weight_3d)

    heating_adjusted = heating + offset

    # Verify energy conservation
    total_after = np.sum(heating_adjusted * weight_3d)

    if verbose:
        print(f"Volume-integrated heating before: {total_before:.6e}")
        print(f"Applied offset: {offset:.6e}")
        print(f"Volume-integrated heating after:  {total_after:.6e}")

    return heating_adjusted, offset

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    """
    Find the distance between two points on Earth
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # distance in kilometers

def get_pfull(data,pressure_levels=np.arange(1000, 0, -100)):
    """
    Process WRF pressure data to be in the same shape as the other 3D variables
    """
    # Shape of my data
    shape = data.reshape(data.shape[0],10,360,208).shape
    # Reshape to (1, 10, 1, 1) to broadcast across the entire (108, 10, 360, 208) shape
    pressure_array = pressure_levels.reshape(1, 10, 1, 1)
    # Broadcast to full shape
    full_array = np.broadcast_to(pressure_array, shape)
    return full_array

def train_valid_test(expvarlist=None,validindex=None,testindex=None,concat='Yes'):
    """
    Seperate data into Train, Validation, and Test based on experiment index numbers
    """
    X_valid, X_test = [expvarlist[i] for i in validindex], [expvarlist[i] for i in testindex]
    X_traint = expvarlist.copy()
    popindex = validindex+testindex
    X_train = [X_traint[i] for i in range(len(X_traint)) if i not in popindex]
    #assert len(X_train)==16, 'wrong train-valid-test separation!'
    if concat=='Yes':
        return np.concatenate([X_train[i] for i in range(len(X_train))],axis=0), np.concatenate([X_valid[i] for i in range(len(X_valid))],axis=0), np.concatenate([X_test[i] for i in range(len(X_test))],axis=0)
    else:
        return X_train, X_valid, X_test

# -------------------- Azimuthal FFT parts --------------------
def theta_fft_components(Q, axisTHETA=1):
    """Return A0, A(m>=1), B(m>=1) for Q(z,theta,r)."""
    Nz, Nth, Nr = Q.shape # Shape of the input variable
    F = np.fft.rfft(Q, axis=axisTHETA) # Perform FFT on the azimuthal axis
    M = F.shape[axisTHETA]-1 # We only need the m>=1 components, so the shape is half of the original array
    A0 = F[:,0,:].real / Nth # Amplitude (0th components)
    scale = 2.0/Nth 
    A = np.zeros((M, Nr, Nz)); B = np.zeros((M, Nr, Nz))
    for m in range(1, M+1):
        Fm = np.transpose(F[:,m,:])
        A[m-1] = scale*Fm.real # Re(FFT) = Nth/2 * Am
        B[m-1] = -scale*Fm.imag # Im(FFT) = -Nth/2 * Bm
    return A0, A, B

def theta_ifft_components(A0, A, B, Ntheta):
    """
    Inverse Fourier Transform to retrieve the orignal signal
    """
    A0 = A0.transpose() #FFT outputs is (z,r), change it to (r,z)
    Nr, Nz = A0.shape 
    M = A.shape[0]
    th = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
    Q = np.zeros((Nr,Nz,Ntheta))
    Q += A0[:,:,None]
    for m in range(1, M+1):
        Q += A[m-1][:,:,None]*np.cos(m*th)[None,None,:]
        Q += B[m-1][:,:,None]*np.sin(m*th)[None,None,:]
    return Q
    