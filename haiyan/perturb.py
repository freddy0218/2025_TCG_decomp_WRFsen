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
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from netCDF4 import Dataset
from wrf import getvar, CoordPair, xy_to_ll, ll_to_xy, get_cartopy, latlon_coords
from wrf import to_np

from wrf import getvar, interplevel
# list(proc_U['pres'].data)
def compute_tc_shear(u, v, preslv, x0, y0, dx_km=3.0,
                     inner_km=200.0, outer_km=800.0,
                     p_top=200.0, p_bot=850.0):
    """
    Compute mean 200–850 hPa shear vector in a TC-centered annulus.

    Parameters
    ----------
    ds : netCDF Dataset (opened with netCDF4 or xarray + wrf-python)
        WRF output file handle
    x0, y0 : int
        TC center grid indices (x,y) in WRF domain
    dx_km : float
        Grid spacing (km)
    inner_km, outer_km : float
        Inner and outer radius of annulus (km)
    p_top, p_bot : float
        Top and bottom pressure levels (hPa)

    Returns
    -------
    shear_u, shear_v : floats
        Zonal and meridional shear components (m/s)
    shear_mag : float
        Shear magnitude (m/s)
    """

    # --- Interpolate to levels
    u200 = u[preslv.index(p_top),...]
    v200 = v[preslv.index(p_top),...]
    u850 = u[preslv.index(p_bot),...]
    v850 = v[preslv.index(p_bot),...]

    # --- Make distance mask in km
    ny, nx = u200.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    dx = (X - x0) * dx_km
    dy = (Y - y0) * dx_km
    r = np.sqrt(dx**2 + dy**2)

    mask = (r >= inner_km) & (r <= outer_km)

    # --- Area-average wind in annulus
    u200m = np.nanmean(to_np(u200.where(mask)))
    v200m = np.nanmean(to_np(v200.where(mask)))
    u850m = np.nanmean(to_np(u850.where(mask)))
    v850m = np.nanmean(to_np(v850.where(mask)))

    # --- Shear vector
    shear_u = u200m - u850m
    shear_v = v200m - v850m
    shear_mag = np.sqrt(shear_u**2 + shear_v**2)

    return shear_u, shear_v, shear_mag

def change_time_resolution_interpolate(times_1hr, times_3min, t0, array=None):
    """
    Interpolates time series data from 1-hour resolution to 3-minute resolution.
    
    Handles data shaped like:
      - (time,) scalars
      - list of vectors, some entries NaN (inhomogeneous)
    
    Returns array of shape (len(times_3min), n_features).
    """
    # Convert times to numeric (seconds since t0)
    t_sec_1hr = (times_1hr - t0).total_seconds().values
    t_sec_3min = (times_3min - t0).total_seconds().values

    # --- Normalize input to 2D array ---
    # Find maximum feature length
    max_len = max(len(x) if isinstance(x, (list, np.ndarray)) else 1
                  for x in array)
    
    arr = np.full((len(array), max_len), np.nan)
    for i, row in enumerate(array):
        if isinstance(row, (list, np.ndarray)):
            arr[i, :len(row)] = row
        elif np.isnan(row):   # scalar nan
            continue
        else:
            arr[i, 0] = row

    n_time, n_feat = arr.shape
    result = np.full((len(t_sec_3min), n_feat), np.nan)

    # --- Interpolate each feature independently ---
    for j in range(n_feat):
        series = arr[:, j]
        valid_mask = ~np.isnan(series)
        if np.sum(valid_mask) < 2:
            continue  # not enough data to interpolate

        t_valid = t_sec_1hr[valid_mask]
        series_valid = series[valid_mask]

        f = interp1d(t_valid, series_valid, kind="linear", bounds_error=False)
        series_3min = f(t_sec_3min)

        # Mask outside valid range
        t_min, t_max = t_valid.min(), t_valid.max()
        series_3min[(t_sec_3min < t_min) | (t_sec_3min > t_max)] = np.nan

        result[:, j] = series_3min

    return result

def gaussian_radial(r, r0, rmax=None):
    """
    Gaussian core with optional outer hard taper at rmax.
    r0 sets the e-folding radius of each lobe.
    """
    R = np.exp(-(r**2)/(2*r0**2))
    if rmax is not None:
        R = np.where(r <= rmax, R, 0.0)
    return R


def compact_cosine_radial(r, r_in, r_out):
    """
    Smooth C^1 compact support: flat inside r_in, tapers to 0 by r_out.
    """
    R = np.zeros_like(r)
    core = r <= r_in
    ann  = (r > r_in) & (r < r_out)
    R[core] = 1.0
    s = (r[ann]-r_in)/(r_out-r_in)
    R[ann] = 0.5*(1+np.cos(np.pi*s))  # goes to 0 at r_out
    return R


def u_shaped_vertical(z, zmid, zsig, low=1.0, high=1.0, minval=0.2):
    # Gaussian gives max=1 at zmid, we flip it so it's min at zmid
    base = np.exp(-((z - zmid)**2)/(2*zsig**2))  
    u = minval + (1 - minval) * (1 - base)   # min at zmid, max away
    z0, z1 = float(z.min()), float(z.max())
    weight_lr = (z1 - z) / (z1 - z0) * low + (z - z0) / (z1 - z0) * high
    return u * weight_lr

def wavenumber1_field(x, y, z, x0, y0,
                      amp=1.0,
                      amp_profile=None,   # <— NEW: shape (nz,)
                      phi0_deg=0.0,
                      a=1.0, b=1.0,
                      radial_kind='gaussian',
                      r0=50e3, rmax=None,
                      r_in=None, r_out=None,
                      vertical_kind='u_shaped',  # 'u_shaped' | 'flat'
                      zmid=None, zsig=None,
                      low=1.0, high=1.0, minval=0.2):

    phi0 = np.deg2rad(phi0_deg)

    x_is_1d = (np.ndim(x) == 1)
    y_is_1d = (np.ndim(y) == 1)
    z_is_1d = (np.ndim(z) == 1)

    if x_is_1d and y_is_1d and z_is_1d:
        X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
        out_dims = ('z','y','x')
        transpose_order = (2,0,1)
    else:
        X, Y, Z = np.broadcast_arrays(x, y, z)
        out_dims = None
        transpose_order = None

    Xp = X - x0
    Yp = Y - y0

    re = np.sqrt((Xp/a)**2 + (Yp/b)**2)
    theta = np.arctan2(Yp, Xp)

    # --- radial
    if radial_kind == 'gaussian':
        R = gaussian_radial(re, r0, rmax=rmax)
    elif radial_kind == 'compact':
        if r_in is None or r_out is None:
            raise ValueError("compact radial requires r_in and r_out")
        R = compact_cosine_radial(re, r_in, r_out)
    else:
        raise ValueError("radial_kind must be 'gaussian' or 'compact'.")

    # --- vertical
    if vertical_kind == 'u_shaped':
        if (zmid is None) or (zsig is None):
            zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            zmid = 0.5*(zmin+zmax) if zmid is None else zmid
            zsig = 0.25*(zmax-zmin) if zsig is None else zsig
        if z_is_1d:
            Vz = u_shaped_vertical(z, zmid, zsig, low=low, high=high, minval=minval)
            V = Vz[np.newaxis, np.newaxis, :] if (x_is_1d and y_is_1d) else Vz
        else:
            V = u_shaped_vertical(Z, zmid, zsig, low=low, high=high, minval=minval)
    elif vertical_kind == 'flat':
        V = 1.0  # no vertical shaping
    else:
        raise ValueError("vertical_kind must be 'u_shaped' or 'flat'.")

    # --- azimuthal
    Atheta = np.cos(theta - phi0)
    base_pattern = R * Atheta * V

    # --- amplitude control
    if amp_profile is not None:
        amp_profile = np.asarray(amp_profile)
        if z_is_1d:
            if amp_profile.shape != (len(z),):
                raise ValueError("amp_profile must have shape (len(z),)")
            scale = amp_profile[np.newaxis, np.newaxis, :] if (x_is_1d and y_is_1d) else amp_profile
        else:
            # If Z has physical heights/etas, interpolate profile onto Z
            scale = np.interp(Z, z, amp_profile)
        P_xyz = base_pattern * scale
    else:
        P_xyz = amp * base_pattern

    if x_is_1d and y_is_1d and z_is_1d:
        P = xr.DataArray(P_xyz.transpose(transpose_order),
                         coords={'z': z, 'y': y, 'x': x},
                         dims=out_dims,
                         name='wn1_perturbation')
    else:
        P = xr.DataArray(P_xyz, name='wn1_perturbation')

    return P, R, V