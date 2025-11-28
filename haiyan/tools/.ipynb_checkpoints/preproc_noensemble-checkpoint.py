import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import xarray as xr
import numpy as np
import seaborn as sns
import glob,os,sys
from tqdm import tqdm
import datetime
from netCDF4 import Dataset
import json,pickle
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter
from tools import derive_var,read_and_proc
import gc

##############################################################################################################
# Numerical
##############################################################################################################
def nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx.values

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
        return np.asarray(result)

##############################################################################################################
# PCA
##############################################################################################################
def do_PCA(var=None,timezoom=None,smooth='Yes',gaussian=0.9,fromcenter='Yes',inradius=None,outradius=None,donormal='Yes',do_PCA='Yes',do_center='Yes'):
    """
    Var: Input variable (must have 4 dimensions! Time-pres-theta-radius)
    """
    if smooth=='Yes':
        from scipy.ndimage import gaussian_filter
        normal_var = []
        if donormal=='Yes':
            for presindex in range(len(var[0,:,0,0])):
                normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'Yes'),sigma=gaussian))
        elif donormal=='No':
            if do_center=='Yes':
                for presindex in range(len(var[0,:,0,0])):
                    normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'No','Yes'),sigma=gaussian))
            elif do_center=='No':
                for presindex in range(len(var[0,:,0,0])):
                    normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'No','No'),sigma=gaussian))  
        normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    else:
        normal_var = []
        for presindex in range(len(var[0,:,0,0])):
            normal_var.append(normalize_inner(var[:,presindex,:,:],outradius))
        normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    if fromcenter=='Yes':
        normal_varf = np.asarray([normal_var[timezoom[0]:timezoom[1]][i,:,:,:outradius].flatten() \
                                  for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])
    elif fromcenter=='No':
        normal_varf = np.asarray([normal_var[timezoom[0]:timezoom[1]][i,:,:,inradius:outradius].flatten() \
                                  for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])        
    #print("--Finish preprocesing--")
    if do_PCA=='Yes':
        from sklearn.decomposition import PCA
        import time
        start_time = time.time()
        skpcaVAR = PCA()
        skpcaVAR.fit(normal_varf.copy())
        #print("--- %s seconds ---" % (time.time() - start_time))
        return skpcaVAR,normal_var,normal_varf
    else:
        return normal_var,normal_varf

def normalize_inner(var=None,outerradius=None,standard='Yes',docenter='No'):
    PWper_ctrl = []
    for indx in range(len(var[:,0,0])):
        if docenter=='Yes':
            temp = var[indx,:,:outerradius]-np.nanmean(var[indx,:,:outerradius],axis=(0,1))
        elif docenter=='No':
            temp = var[indx,:,:outerradius]
        if standard=='Yes':
            PWper_ctrl.append((temp-np.nanmean(temp))/np.nanstd(temp))
        elif standard=='No':
            PWper_ctrl.append((temp))
    del temp
    return np.asarray(PWper_ctrl)

##############################################################################################################
# Preprocess - original PCA
##############################################################################################################
def do_uradvtan_nonensemble(originpath=None,coor_pres=None,expname=None,savepath=None):
    """
    Derive urad/vtan and save to /scratch
    """
    ctrlvar_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+str(expname)+'/azim_U_*')[0],\
                                                     glob.glob(originpath+str(expname)+'/azim_V_*')[0]],fieldname=['U','V'])
    r500 = nearest_index(ctrlvar_dict['U']['U'].radius,500)
    ctrlhUrad,ctrlhVtan,_ = derive_var.ruppert_vtmax_calc(ctrlvar_dict['U']['U'].data,\
                                                          ctrlvar_dict['V']['V'].data,r500,ctrlvar_dict['U']['U'][:,0,:,:].azmiuth.values,None)
    read_and_proc.save_to_pickle(savepath+'/'+str(expname)+'/'+'urad',ctrlhUrad,'PICKLE')
    read_and_proc.save_to_pickle(savepath+'/'+str(expname)+'/'+'vtan',ctrlhVtan,'PICKLE')
        
    del ctrlvar_dict,ctrlhUrad,ctrlhVtan
    gc.collect()
    return None  



def save_PCA(var=None,varname=['U','V','W','QV','dthdt'],savepath=None):
    ###################################################################
    # Do PCA
    #CTRL_PCAdict,CTRL_flatvardict,CTRL_origvardict = {},{},{}
    for indx,obj in tqdm(enumerate(var)):
        temp1,temp3,temp2 = do_PCA(var=obj,timezoom=[0,-1],smooth='Yes',
                                   gaussian=[0,0,0],fromcenter='Yes',inradius=None,outradius=r500,donormal='No',do_PCA='Yes',do_center='No')
        read_and_proc.save_to_pickle(savepath+'pca/'+str(varname[indx])+'_pca',temp1)
        read_and_proc.save_to_pickle(savepath+'origvar/'+str(varname[indx])+'_origvar',temp3)
        read_and_proc.save_to_pickle(savepath+'flatvar/'+str(varname[indx])+'_flatvar',temp2) 
        del temp1,temp2,temp3
        gc.collect()        
    return None

##############################################################################################################
# Preprocess - for sensitivity tests
##############################################################################################################
def preproc_onevar(varname=None,originpath=None,expname=None,sigma=[3,0,0,0],addsenmethod='new'):
    ###############################################################################################
    # var
    ###############################################################################################
    var_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+str(expname)+'/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
    #ctrl_t2,ctrl_t4 = nearest_index(ctrlvar_dict['W']['W'].time/24,0.5)-1,nearest_index(ctrlvar_dict['W']['W'].time/24,7)-1
    r500=nearest_index(var_dict[varname][varname].radius,500)
    ###############################################################################################
    # dtheta
    ###############################################################################################
    if expname=='ctl':
        varL = var_dict[varname][varname]
    elif (expname=='ncrf_36h') or (expname=='lwcrf'):
        ctrlvar_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+'ctl/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
        if addsenmethod=='orig':
            varL = read_and_proc.add_ctrl_before_senstart(ctrlvar_dict[varname][varname],var_dict[varname][varname],'NCRF36','Yes')
        elif addsenmethod=='new':
            varL = read_and_proc.add_ctrl_before_senstart_ctrlbase(ctrlvar_dict[varname][varname],var_dict[varname][varname],'NCRF36','Yes')
        del ctrlvar_dict
        gc.collect()
    elif (expname=='ncrf_60h'):
        ctrlvar_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+'ctl/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
        if addsenmethod=='orig':
            varL = read_and_proc.add_ctrl_before_senstart(ctrlvar_dict[varname][varname],var_dict[varname][varname],'NCRF60','Yes')
        elif addsenmethod=='new':
            varL = read_and_proc.add_ctrl_before_senstart_ctrlbase(ctrlvar_dict[varname][varname],var_dict[varname][varname],'NCRF60','Yes')
        del ctrlvar_dict
        gc.collect()
    ###############################################################################################
    # smooth
    ###############################################################################################
    if expname=='ctl':
        var_s = gaussian_filter(varL,sigma=sigma)
        del varL,var_dict
    else:
        var_s = gaussian_filter(varL,sigma=sigma)
        del varL,var_dict    
    gc.collect()
    var_s = np.nan_to_num(var_s)
    print("---Finish!---")
    return var_s

def preproc_dudvdw(uvtpath=None,originpath=None,expname=None):
    ###############################################################################################
    # urad,vtan,theta & swapaxes
    ###############################################################################################
    urad = np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+str(expname)+'/urad')),0,1)
    vtan = np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+str(expname)+'/vtan')),0,1)
    ###############################################################################################
    # w
    ###############################################################################################
    var_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+str(expname)+'/azim_W*')[0]],fieldname=['W'])
    r500=nearest_index(var_dict['W']['W'].radius,500)
    ###############################################################################################
    # dtheta
    ###############################################################################################
    if expname=='ctl':
        durad = forward_diff(gaussian_filter(urad,sigma=[3,0,0,0]),60*60,0)
        dvtan = forward_diff(gaussian_filter(vtan,sigma=[3,0,0,0]),60*60,0)
        dw = forward_diff(gaussian_filter(var_dict['W']['W'],sigma=[3,0,0,0]),60*60,0)
    elif (expname=='ncrf_36h') or (expname=='lwcrf'):
        ctrl_uradA = np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+'/ctl/urad')),0,1)
        ctrl_vtanA = np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+'/ctl/vtan')),0,1)
        uradL = read_and_proc.add_ctrl_before_senstart(ctrl_uradA,urad,'ncrf_36h','Yes')
        vtanL = read_and_proc.add_ctrl_before_senstart(ctrl_vtanA,vtan,'ncrf_36h','Yes')
        durad,dvtan=forward_diff(gaussian_filter(uradL,sigma=[3,0,0,0]),60*60,0),forward_diff(gaussian_filter(vtanL,sigma=[3,0,0,0]),60*60,0)
        del ctrl_uradA,ctrl_vtanA,uradL,vtanL
        gc.collect()
        ctrlvar_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+'ctl/azim_W*')[0]],fieldname=['W'])
        wL = read_and_proc.add_ctrl_before_senstart(ctrlvar_dict['W']['W'],var_dict['W']['W'],'ncrf_36h','Yes')
        dw = forward_diff(gaussian_filter(wL,sigma=[3,0,0,0]),60*60,0)
        del ctrlvar_dict,wL
        gc.collect()
    elif (expname=='ncrf_60h'):
        ctrl_uradA = np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+'/ctl/urad')),0,1)
        ctrl_vtanA = np.swapaxes(np.asarray(read_and_proc.depickle(uvtpath+'/ctl/vtan')),0,1)
        uradL = read_and_proc.add_ctrl_before_senstart(ctrl_uradA,urad,'ncrf_60h','Yes')
        vtanL = read_and_proc.add_ctrl_before_senstart(ctrl_vtanA,vtan,'ncrf_60h','Yes')
        durad,dvtan=forward_diff(gaussian_filter(uradL,sigma=[3,0,0,0]),60*60,0),forward_diff(gaussian_filter(vtanL,sigma=[3,0,0,0]),60*60,0)
        del ctrl_uradA,ctrl_vtanA,uradL,vtanL
        gc.collect()
        ctrlvar_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+'ctl/azim_W*')[0]],fieldname=['W'])
        wL = read_and_proc.add_ctrl_before_senstart(ctrlvar_dict['W']['W'],var_dict['W']['W'],'ncrf_60h','Yes')
        dw = forward_diff(gaussian_filter(wL,sigma=[3,0,0,0]),60*60,0)        
        del ctrlvar_dict,wL
        gc.collect()
    durad = np.nan_to_num(durad)
    dvtan = np.nan_to_num(dvtan)
    dw = np.nan_to_num(dw)
    print("---Finish!---")
    return durad,dvtan,dw

def save_onevar(var=None,varname=None,savepath=None,originpath=None):
    ###################################################################
    var_dict = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(originpath+'ctl/azim_W'+'_*')[0]],fieldname=['W'])
    #ctrl_t2,ctrl_t4 = nearest_index(ctrlvar_dict['W']['W'].time/24,0.5)-1,nearest_index(ctrlvar_dict['W']['W'].time/24,7)-1
    r500=nearest_index(var_dict['W']['W'].radius,500)
    del var_dict
    gc.collect()
    # Do PCA
    #CTRL_PCAdict,CTRL_flatvardict,CTRL_origvardict = {},{},{}
    for indx,obj in tqdm(enumerate(var)):
        _,temp2 = do_PCA(var=obj,timezoom=[24,120],smooth='Yes',
                           gaussian=[0,0,0],fromcenter='Yes',inradius=None,outradius=r500,donormal='No',do_PCA='No',do_center='No')
        #read_and_proc.save_to_pickle(savepath+'/heat/'+str(varname[indx])+'_pca',temp1)
        read_and_proc.save_to_pickle(savepath+str(varname[indx])+'_flatvar',temp2) 
        del temp2
        gc.collect()        
    return None