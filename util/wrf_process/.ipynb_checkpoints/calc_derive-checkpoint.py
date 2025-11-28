import numpy as np
import sys
from scipy import ndimage
import sys
from scipy import spatial
from wrf import ll_to_xy

def relvort(u, v, lat, lon):
    
    a = 6371e3 # Earth radius, m
    deg2rad = np.pi/180
    deg2meters = a * deg2rad
    try:
        cosf = np.cos(np.radians(lat)).data
    except:
        cosf = np.cos(np.radians(lat))

    dudy = np.gradient( u , lat*deg2meters , axis=1)
    dvdx = np.gradient( v , lon*deg2meters , axis=2) / cosf[np.newaxis,:,np.newaxis]

    # print("Shape of gradient variable:",np.shape(dvdx))
    vor = (dvdx - dudy)

    return vor

def standardization(data):
    return (data-np.nanmean(data))/np.nanstd(data)

def smooth_var(f,nx_sm,nx_repeat,nt_smooth):
    # Smooth input variable in x,y
    f_smooth = ndimage.uniform_filter(f,size=(0,nx_sm,nx_sm),mode='nearest')
    for ido in range(nx_repeat-1):
        f_smooth = ndimage.uniform_filter(f_smooth,size=(0,nx_sm,nx_sm),mode='nearest')
    # Smooth input variable in time
    if nt_smooth==0:
        return f_smooth
    else:
        f_smooth = ndimage.uniform_filter(f_smooth,size=(nt_smooth,0,0),mode='nearest')
        return f_smooth