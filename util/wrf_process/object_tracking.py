# 
# Track a TC or precursor vortex using an object-based algorithm, following
#   Davis et al. (2006, MWR) and Rios-Berrios et al. (2018, JAS).
# 
# The sell of this approach is that any 2D variable (vorticity, MSLP, precip)
#   can be used as input.
# 
# Steps:
#   1) Large-scale smoothing XX in time and space.
#   2) Top-5% of variable is retained.
#   3) Centroid is found via weighted integral approach.
#  xxx 4) Sanity check for maximum possible phase speed for continuity.
# 
# Input:
#   f   = input variable assumed to be in form f = f(t,y,x)
#   lon = longitude points (deg) as lon = lon(y,x)
#   lat = longitude points (deg) as lat = lat(y,x)
#   sens_test = True or False, for if this case is a sensitivity test and
#       should have its tracking initiated on the basis of another simulation
#   basis = [lon, lat] the x,y location from which to begin tracking forward in
#       time, which is only used in the case of sens_test = True
# 
# Returns: numpy array[itrack,2] where itrack corresponds to (potentially)
#   multiple identified tracks and the second dimension is (lon,lat).
# 
# James Ruppert
# June 2022
# 

import numpy as np
from scipy import ndimage
import sys
from scipy import spatial
from wrf import ll_to_xy

def object_track(f, lon, lat, sens_test, basis, nx_sm, nx_repeat, nt_smooth, r_max):

    shape=np.shape(f)
    nt,ny,nx = shape

    if len(shape) != 3:
        print("Check input dimensions!")
        print("Shape: ",shape)
        sys.exit()

    #############################################

    # SETTINGS
    # 3-dimensional lon/lat for weighting
    lon3d = np.repeat(lon[np.newaxis,:,:], nt, axis=0)
    lat3d = np.repeat(lat[np.newaxis,:,:], nt, axis=0)

    #############################################

    # SMOOTHING

    # Smooth input variable in x,y
    f_smooth = ndimage.uniform_filter(f,size=(0,nx_sm,nx_sm),mode='nearest')
    for ido in range(nx_repeat-1):
        f_smooth = ndimage.uniform_filter(f_smooth,size=(0,nx_sm,nx_sm),mode='nearest')

    # Smooth input variable in time
    # for ido in range(nt_repeat):
    f_smooth = ndimage.uniform_filter(f_smooth,size=(nt_smooth,0,0),mode='nearest')

    # BULK MASKING

    # Mask out values < 3 sigma
    f_sigma = f_smooth / np.std(f_smooth)
    f_masked = np.ma.array(f_sigma)
    f_masked = np.ma.masked_where(np.abs(f_masked) < 3, f_masked, copy=False)

    # Mask out data within 0.5*r_max from boundaries
    f_masked = np.ma.masked_where(lon3d <= lon[0,0]   +0.5*r_max, f_masked, copy=False)
    f_masked = np.ma.masked_where(lon3d >= lon[0,nx-1]-0.5*r_max, f_masked, copy=False)
    f_masked = np.ma.masked_where(lat3d <= lat[0,0]   +0.5*r_max, f_masked, copy=False)
    f_masked = np.ma.masked_where(lat3d >= lat[ny-1,0]-0.5*r_max, f_masked, copy=False)

    #############################################

    if sens_test:

        # Assuming a sensitivity test restarted from another simulation
        print('Assuming sensitivity test; using basis to initialize track')

        # Therefore, using that restart time step as the basis from which
        # to track forward in time.

        radius = np.sqrt( (lon-basis[0])**2 + (lat-basis[1])**2 )
        itmax = 0

    else:
    
        # Assuming a cold start; will identify all-time-max object magnitude
        # and track forward and backward from there.
        print('Assuming cold start and using no basis')

        # Mask out first time step
        f_masked.mask[0,:,:] = True

        # Locate the all-time maximum value
        fmax = np.max(f_masked)
        mloc=np.where(f_masked == fmax)
        itmax = mloc[0][0]
        xmax=mloc[2][0]
        ymax=mloc[1][0]

        radius = np.sqrt( (lon-lon[ymax,xmax])**2 + (lat-lat[ymax,xmax])**2 )

    # Mask beyond specified radius at 2 neighboring time steps
    for it in range( np.maximum(itmax-1,0) , np.minimum(itmax+1,nt-1)+1 ):

        f_masked[it,:,:] = np.ma.masked_where(radius > r_max, f_masked[it,:,:], copy=False)

    # Do the same iterating all the way backward from itmax
    for it in range( np.maximum(itmax-1,0), 0, -1):

        fmax = np.max(f_masked[it,:,:])
        mloc = np.where(f_masked[it,:,:] == fmax)
        xmax = mloc[1][0]
        ymax = mloc[0][0]
        
        radius = np.sqrt( (lon-lon[ymax,xmax])**2 + (lat-lat[ymax,xmax])**2 )
        f_masked[it-1,:,:] = np.ma.masked_where(radius > r_max, f_masked[it-1,:,:], copy=False)

    # Do the same iterating all the way forward from itmax
    for it in range(itmax+1,nt-1):

        fmax = np.max(f_masked[it,:,:])
        mloc = np.where(f_masked[it,:,:] == fmax)
        xmax = mloc[1][0]
        ymax = mloc[0][0]

        radius = np.sqrt( (lon-lon[ymax,xmax])**2 + (lat-lat[ymax,xmax])**2 )
        f_masked[it+1,:,:] = np.ma.masked_where(radius > r_max, f_masked[it+1,:,:], copy=False)

    #############################################

    # TRACKING

    # Track maxima in time as the centroid = mean of latitude/longitude weighted by f
    clon = np.average(lon3d,axis=(1,2),weights=f_masked)
    clat = np.average(lat3d,axis=(1,2),weights=f_masked)

    track = np.ma.concatenate([clon[np.newaxis,:],clat[np.newaxis,:]])

    return track, f_masked

# Function to account for crossing of the Intl Date Line
def dateline_lon_shift(lon_in, reverse):
    if reverse == 0:
        lon_offset = np.zeros(lon_in.shape)
        lon_offset[np.where(lon_in < 0)] += 360
    else:
        lon_offset = np.zeros(lon_in.shape)
        lon_offset[np.where(lon_in > 180)] -= 360
    # return lon_in + lon_offset
    return lon_offset

def tree_latlon_to_xy(ncfile,longitude,latitude,wantpointlong,wantpointlat):
    tree = spatial.KDTree(list(zip(longitude.ravel(), 
                                   latitude.ravel()
                                  )
                              )
                         )
    
    wantpointlist = list(zip(wantpointlong,
                             wantpointlat)
                        )
    point_dist, point_idx = tree.query(wantpointlist)
    want_latlon = [tree.data[obj] for obj in point_idx]
    latlon_xy = [ll_to_xy(ncfile,zobj[1],zobj[0]).data for zobj in want_latlon]
    return {'tree':tree,'point_dist':point_dist,'point_idx':point_idx,'want_latlon':want_latlon,'latlon_xy':latlon_xy}
