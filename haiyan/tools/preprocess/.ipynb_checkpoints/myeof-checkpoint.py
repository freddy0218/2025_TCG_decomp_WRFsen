import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter

##############################################################################################################################################
# Preprocessing fields
def preproc_fields(var=None,timezoom=None,smooth='Yes',gaussian=0.9,fromcenter='Yes',inradius=None,outradius=None,dostandard='Yes'):
    """
    Var: Input variable (must have 4 dimensions! Time-pres-theta-radius)
    """
    if smooth=='Yes':
        from scipy.ndimage import gaussian_filter
        normal_var = []
        if dostandard=='Yes':
            for presindex in range(len(var[0,:,0,0])):
                normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'Yes'),sigma=gaussian))
        elif dostandard=='No':
            for presindex in range(len(var[0,:,0,0])):
                normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'No'),sigma=gaussian))            
            normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    else:
        normal_var = []
        for presindex in range(len(var[0,:,0,0])):
            normal_var.append(normalize_inner(var[:,presindex,:,:],outradius))
            normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    if fromcenter=='Yes':
        normal_varf = np.asarray([normal_var[i,:,:,:outradius].flatten() for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])
    elif fromcenter=='No':
        normal_varf = np.asarray([normal_var[i,:,:,inradius:outradius].flatten() for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])       
    print("--Finish preprocessing--")
    return normal_var,normal_varf

def do_PCA(var=None,timezoom=None,smooth='Yes',gaussian=[0,0,0],fromcenter='Yes',inradius=None,outradius=None,donormal='No',do_PCA='Yes',do_center='No'):
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
                    normal_var.append(normalize_inner(var[:,presindex,:,:],outradius,'No','No'))  
        normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    else:
        normal_var = []
        for presindex in range(len(var[0,:,0,0])):
            normal_var.append(normalize_inner(var[:,presindex,:,:],outradius,'No','No'))
        normal_var = np.swapaxes(np.asarray(normal_var),0,1)
    if fromcenter=='Yes':
        normal_varf = np.asarray([normal_var[timezoom[0]:timezoom[1]][i,:,:,:outradius].flatten() \
                                  for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])
    elif fromcenter=='No':
        normal_varf = np.asarray([normal_var[timezoom[0]:timezoom[1]][i,:,:,inradius:outradius].flatten() \
                                  for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])        
    if do_PCA=='Yes':
        from sklearn.decomposition import PCA
        import time
        start_time = time.time()
        skpcaVAR = PCA()
        skpcaVAR.fit(normal_varf.copy())
        return skpcaVAR,normal_var,normal_varf
    else:
        return normal_var,normal_varf

def normalize_inner(var=None,outerradius=None,standard='No',docenter='No'):
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

def proc_invar_forPCA(var=None,timezoom=None,smooth='Yes',gaussian=[0,0,0],fromcenter='Yes',inradius=None,outradius=None,donormal='No',doPCA='No'):
    return do_PCA(var=var,timezoom=timezoom,smooth='Yes',gaussian=[0,0,0],fromcenter='Yes',inradius=None,outradius=None,donormal='No',doPCA='No')

##############################################################################################################################################
# I/O 
def standardize(t,formula='II'):
    if formula=='I':
        return (t-np.min(np.asarray(t)))/(np.max(np.asarray(t))-np.min(np.asarray(t)))
    elif formula=='II':
        out = []
        for inte in range(len(t[0,:])):
            out.append((t[:,inte]-np.nanmean(t[:,inte]))/np.nanstd(t[:,inte]))
        return np.asarray(out).T
        #return (t-np.nanmean(np.asarray(t)))/(np.nanstd(np.asarray(t)))
        
def standardize_sen(t,formula='II',basePC=None):
    if formula=='I':
        return (t-np.min(np.asarray(t)))/(np.max(np.asarray(t))-np.min(np.asarray(t)))
    elif formula=='II':
        out = []
        for inte in range(len(t[0,:])):
            out.append((t[:,inte]-np.nanmean(basePC[:,inte]))/np.nanstd(basePC[:,inte]))
        return np.asarray(out).T
        #return (t-np.nanmean(np.asarray(t)))/(np.nanstd(np.asarray(t)))
        
def myPCA_projection(pca_dict=None,varname=None,toproj_flatvar=None,orig_flatvar=None):
    pca_orig = pca_dict[varname].transform(orig_flatvar)
    if pca_dict[varname].mean_ is not None:
        orig_mean = pca_dict[varname].mean_
    projvar_transformed = np.dot(toproj_flatvar-np.mean(toproj_flatvar,axis=0),pca_dict[varname].components_.T)
    del orig_mean
    gc.collect()
    return pca_orig, projvar_transformed

def myPCA_projection_senconst(pca_dict=None,varname=None,toproj_flatvar=None,orig_flatvar=None,ctrlvar_store=None,senvar_store=None):
    """
    Output constant terms for sensitivity experiments (np.mean(t_CTRL)-np.mean(t_NCRF))
    """
    def repeat_1d(array1d=None,repeattime=96):
        b = []
        for ind,obj in enumerate(array1d):
            b.append(np.repeat(obj,repeattime))
        return np.asarray(b)
    if pca_dict[varname].mean_ is not None:
        orig_mean = pca_dict[varname].mean_
    projvar_transformed = np.dot(np.mean(ctrlvar_store,axis=0)-np.mean(senvar_store,axis=0),pca_dict[varname].components_.T)
    projvar_transformed2d = repeat_1d(projvar_transformed,96)
    del orig_mean
    gc.collect()
    return projvar_transformed2d
    
def output_PCAtimeseries(PCAdict=None,varname=None,flatvar=None,no_comp=None,standard=True,standardtype='II',TYPE='ctrl',basePC=None):
    """
    Output PC time series for CTRL experiment
    """
    if varname is None:
        varname=['dtheta','u','v','w','qv']
    timeseries_out = {}
    for index,var in enumerate(varname):
        if standard is True:
            if TYPE=='ctrl':
                timeseries_out[var] = standardize(PCAdict[var].transform(flatvar[index])[:,0:no_comp[index]],standardtype) 
            else:
                #timeseries_out[var] = standardize_sen(PCA_dict[var].transform(flatvar[index])[:,0:no_comp[index]],standardtype,basePC=basePC[var])
                timeseries_out[var] = standardize(PCAdict[var].transform(flatvar[index])[:,0:no_comp[index]],standardtype)
        else:
            timeseries_out[var] = PCAdict[var].transform(flatvar[index])[:,0:no_comp[index]]
    return timeseries_out

##############################################################################################################################################
# Reconstruct original field
def reconstruct_fromPCA(PCAdict=None,Afdict=None,Adict=None,component=5,ALL='No'):
		"""
		PCAdict: Processed PCA objects
		Afdict: Flattened processed variable array
		Adict: Processed variable array
		component: Number of component used to reconstruct original array
		ALL: Use all components?
		"""
		if ALL=='Yes':
				Xhat = np.dot(PCAdict.transform(Afdict)[:,:],PCAdict.components_[:,:])	
		else:
				Xhat = np.dot(PCAdict.transform(Afdict)[:,:int(component)],PCAdict.components_[:int(component),:])
				Xhatc = Xhat.copy()
				Xhatc+=np.mean(Afdict, axis=0)		
				TESTrecon = Xhatc.reshape((Xhatc.shape[0],Adict[0].shape[0],Adict[0].shape[1],Adict[0].shape[2]))
		return TESTrecon
