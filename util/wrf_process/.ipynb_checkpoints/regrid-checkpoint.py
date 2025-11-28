import numpy as np
from netCDF4 import Dataset
import xarray as xr
import gc, glob, tqdm
import scipy
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)
    return r, theta

def polar2cart(r, theta):
    y = r * np.cos(theta)
    x = r * np.sin(theta)
    return x, y

def index_coords(data, origin=None):
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_y, origin_x = origin
        if origin_y < 0:
            origin_y += ny
        if origin_x < 0:
            origin_x += nx
    
    x, y = np.meshgrid(np.arange(float(nx))-origin_x,
                       origin_y-np.arange(float(ny)))
    return x, y

def cart_image_to_pol(data, origin=None, Jacobian=False, dr=1, dt=None):
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (ny // 2, nx // 2)
    else:
        origin = list(origin)
        if origin[0] < 0:
            origin[0] += ny
        if origin[1] < 0:
            origin[1] += nx

    x,y = index_coords(data, origin=origin)
    r,theta = cart2polar(x,y)

    nr= int(np.ceil((r.max()-r.min()) / dr))
    if dt is None:
        nt = max(nx,ny)
    else:
        nt = int(np.ceil((theta.max()-theta.min())/dt))

    r_i = np.linspace(r.min(),r.max(),nr,endpoint=False)
    theta_i = np.linspace(theta.min(),theta.max(),nt,endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    X, Y = polar2cart(r_grid, theta_grid)
    rowi = (origin[0]-Y).flatten()
    coli = (X+origin[1]).flatten()
    coords = np.vstack((rowi, coli))

    zi = map_coordinates(data, coords, output=float)
    output = zi.reshape((nr,nt))

    if Jacobian:
        output *= r_i[:, np.newaxis]
    return output, r_grid, theta_grid

def polar2cartesian(outcoords, inputshape, origin):
    """Coordinate transform for converting a polar array to Cartesian coordinates. 
    inputshape is a tuple containing the shape of the polar array. origin is a
    tuple containing the x and y indices of where the origin should be in the
    output array."""
    
    xindex, yindex = outcoords
    x0, y0 = origin
    x = xindex - x0
    y = yindex - y0
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_index = np.round((theta + np.pi) * inputshape[1] / (2 * np.pi))
    return (r,theta_index)

def proc_tocart(polarfield=None,angle=None,twoD=True,standard=False):
    if twoD==True:
        PWnew = [np.asarray(polarfield)[int(np.abs(angle-360).argmin()),:]]
        for i in np.linspace(0,358,359):
            PWnew.append(np.asarray(polarfield)[int(np.abs(angle-i).argmin()),:])
        PWnew = np.swapaxes(np.asarray(PWnew),0,1)
        del i
        
        if standard==True:
            PWnew = (PWnew-np.nanmean(PWnew))/np.nanstd(PWnew)
        else:
            PWnew=PWnew
        test_2cartesian = scipy.ndimage.geometric_transform(PWnew,polar2cartesian,order=0,mode='constant',output_shape =(PWnew.shape[0]*2,PWnew.shape[0]*2),\
                                                            extra_keywords = {'inputshape':PWnew.shape,'origin':(PWnew.shape[0],PWnew.shape[0])})
    return ((test_2cartesian))

