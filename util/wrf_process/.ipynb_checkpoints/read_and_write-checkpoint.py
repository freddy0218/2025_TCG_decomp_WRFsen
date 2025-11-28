from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, interplevel)
import pickle
import juliandate as jd
import datetime,time

def get_basic_domain(ncfile=None):
    # Get the pressure levels
    pres = getvar(ncfile, "pres")
    # Get the sea level pressure
    slp = getvar(ncfile, "slp")
    # Get the latitude and longitude points
    lats, lons = latlon_coords(slp)
    return lons,lats,pres

import pickle

def save_to_pickle(data,savepath):
    with open(savepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def depickle(savepath):
    with open(savepath, 'rb') as handle:
        b = pickle.load(handle)
    return b

def create_juliandates(start_time=None,ref_time=None,totaltimesteps=68*4,minutesdelta=15):
    savetimes = [start_time]
    for i in range(totaltimesteps):#20):
        start_time += datetime.timedelta(minutes=minutesdelta)
        savetimes.append(start_time)
    del i
    gc.collect()
    
    juliandates = []
    for i in range(len(savetimes)):
        mt,reft = savetimes[i],ref_time
        temp = jd.from_gregorian(mt.year,mt.month,mt.day,mt.hour,mt.minute)-\
        jd.from_gregorian(reft.year,reft.month,reft.day,reft.hour,reft.minute)
        juliandates.append(temp)
    return juliandates