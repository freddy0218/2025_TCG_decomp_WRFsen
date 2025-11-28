import numpy as np
import pickle
from tqdm import tqdm
import gc,glob
from copy import deepcopy

def read_forecast_and_output(filelist=None,filelist_aux=None,SIMPLER=True,subtract=True):
    from copy import deepcopy
    OUTPUT = []
    for i in (range(len(filelist))):
        with open(filelist[i],'rb') as f: temp = deepcopy(pickle.load(f))
        if filelist_aux:
            with open(filelist_aux[i],'rb') as f: tempaux = deepcopy(pickle.load(f))
        else:
            tempaux = 0
        
        if SIMPLER is True:
            if subtract is True:
                ans = temp-tempaux
            else:
                ans = temp
            tempTIME = ans.reshape(39,360,167)
        else:
            tempTIME,tempTIME_aux = temp[int(timestep),:].reshape(39,360,167),tempaux[int(timestep),:].reshape(39,360,167)
        OUTPUT.append(tempTIME)
        del temp,tempaux,tempTIME
        gc.collect()
    return OUTPUT

def read_in_forecast(path=None,subpath=None):
    """
    path = '/scratch/08350/tg876493/mlr_output/exp6_N/ctrlbase/all_corr/'
    subpath = ['ctrl60/','recon_36*'] (subfolder,file syntax)
    """
    storedict = {}
    storevar = ['du','dv','dw']
    for indx,obj in (enumerate(storevar)):
        temp = read_forecast_and_output(sorted(glob.glob(path+subpath[0]+str(obj)+subpath[1])),None,True,False)
        storedict[obj] = temp
        del temp
        gc.collect()
    print("__Finish__!")
    return storedict

def read_forecast_makehov(filelist=None,filelist_aux=None,timestep=48,heightlevel=None,substract=True,SIMPLER=True):
    if substract is True:
        OUTPUT = []
        for i in tqdm(range(len(filelist))):
            with open(filelist[i],'rb') as f: temp = pickle.load(f)
            if filelist_aux:
                with open(filelist_aux[i],'rb') as f: tempaux = deepcopy(pickle.load(f))
            else:
                tempaux = 0
            #print("Finish reading")
            if SIMPLER is True:
                if filelist_aux:
                    tempTIME,tempTIME_aux = temp.reshape(39,360,167),tempaux.reshape(39,360,167)
                else:
                    tempTIME = temp.reshape(39,360,167)                    
            else:
                tempTIME,tempTIME_aux = temp[int(timestep),:].reshape(39,360,167),tempaux[int(timestep),:].reshape(39,360,167)
            if filelist_aux:
                tempOUT = np.nanmean(tempTIME[heightlevel,:,:]-tempTIME_aux[heightlevel,:,:],axis=0)
            else:
                tempOUT = np.nanmean(tempTIME[heightlevel,:,:],axis=0)
            OUTPUT.append(tempOUT)
            del temp,tempTIME,tempOUT,f
            gc.collect()
        return OUTPUT         
    else:
        OUTPUT = []
        for i in tqdm(range(len(filelist))):
            with open(filelist[i],'rb') as f:
                temp = pickle.load(f)
            tempTIME = temp[int(timestep),:].reshape(39,360,167)
            tempOUT = np.nanmean(tempTIME[heightlevel,:,:],axis=0)
            OUTPUT.append(tempOUT)
            del temp,tempTIME,tempOUT,f
            gc.collect()
        return OUTPUT

def read_forecast_makeprofile(filelist=None,filelist_aux=None,SIMPLER=True,TYPE='updraft',radius=None,percentile=50,mean='No'):
    from copy import deepcopy
    OUTPUT,OUTPUT_aux = [],[]
    for i in tqdm(range(len(filelist))):
        with open(filelist[i],'rb') as f: temp = deepcopy(pickle.load(f))
        with open(filelist_aux[i],'rb') as f: tempaux = deepcopy(pickle.load(f))
        if SIMPLER is True:
            if TYPE=='updraft':
                ans = temp-tempaux
                ans[ans<0] = np.nan
                tempTIME = ans.reshape(39,360,167)
            elif TYPE=='downdraft':
                ans = temp-tempaux
                ans[ans>0] = np.nan
                tempTIME = ans.reshape(39,360,167)
            else:
                ans = temp-tempaux
                tempTIME = ans.reshape(39,360,167)
        else:
            tempTIME,tempTIME_aux = temp[int(timestep),:].reshape(39,360,167),tempaux[int(timestep),:].reshape(39,360,167)
        if mean=='No':
            CFAD = np.nanpercentile(tempTIME[:,:,radius[0]:radius[1]],percentile,axis=(1,2))
        else:
            CFAD = np.nanmean(tempTIME[:,:,radius[0]:radius[1]],axis=(1,2))
        #CFAD = make_cfad(inputVar=tempTIME[:,:,radius[0]:radius[1]], desireBIN=desireBIN, \
        #                 bdwth=None, case='reconstruct',creator='numpy',kernal='gaussian')
        OUTPUT.append(CFAD)
        del temp,tempaux,tempTIME
        gc.collect()
    return OUTPUT

def read_forecast_and_output(filelist=None,filelist_aux=None,SIMPLER=True,subtract=True):
    from copy import deepcopy
    OUTPUT = []
    for i in (range(len(filelist))):
        with open(filelist[i],'rb') as f: temp = deepcopy(pickle.load(f))
        if filelist_aux:
            with open(filelist_aux[i],'rb') as f: tempaux = deepcopy(pickle.load(f))
        else:
            tempaux = 0
        
        if SIMPLER is True:
            if subtract is True:
                ans = temp-tempaux
            else:
                ans = temp
            tempTIME = ans.reshape(39,360,167)
        else:
            tempTIME,tempTIME_aux = temp[int(timestep),:].reshape(39,360,167),tempaux[int(timestep),:].reshape(39,360,167)
        OUTPUT.append(tempTIME)
        del temp,tempaux,tempTIME
        gc.collect()
    return OUTPUT