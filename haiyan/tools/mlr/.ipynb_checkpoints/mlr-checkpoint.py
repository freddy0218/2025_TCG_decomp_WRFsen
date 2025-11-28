import numpy as np
from sklearn.linear_model import LinearRegression, Lasso,MultiTaskLassoCV
#from celer import MultiTaskLassoCV
#from celer import Lasso
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import os,sys
#sys.path.insert(1, '/work/08350/tg876493/stampede2/python_codes/2020_TC_CRF/dev/freddy0218/')
from tools import derive_var,read_and_proc
from sklearn.base import BaseEstimator, TransformerMixin
import gc,glob
import pickle

def nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx.values
def save_to_pickle(loc=None,var=None):
    with open(loc,"wb") as f:
        pickle.dump(var,f)
    return None

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
        return np.asarray(result)

def flatten(t):
    #https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    return [item for sublist in t for item in sublist]
def standardize(t,formula='I'):
    if formula=='I':
        return (t-np.min(np.asarray(t)))/(np.max(np.asarray(t))-np.min(np.asarray(t)))
    elif formula=='II':
        return (t-np.mean(np.asarray(t)))/(np.std(np.asarray(t)))
    
def produce_regscore(inPUT=None,outPUT=None,aux_inPUT=None,outtype='score',do_aux=False,
                     algorithm='linear',lassoparam=dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplelassoalpha=0.001):
    """
    inPUT: input data to train model
    outPUT: output data to train model
    aux_inPUT: use input data that is different from original input data for forecast purposes (akin to sensitivity experiments)
    outtype: score=R^2, predict=prediction (Y), aux_predict=prediction-intercept, coeff:coefficient (A)
    do_aux: Use auxiliary input or not
    algorithm: linear regression (linear) / LASSO (lasso) / multilasso
    """
    if algorithm=='linear':
        reg = LinearRegression().fit(inPUT, outPUT)
    elif algorithm=='lasso':
        #reg = MultiTaskLassoCV(**lassoparam).fit(inPUT,outPUT)
        reg = Lasso(simplelassoalpha).fit(inPUT,outPUT)
    if outtype=='score':
        if do_aux is False:
            return reg.score(inPUT, outPUT)
        elif do_aux is True:
            return reg.score(aux_inPUT,outPUT)
    elif outtype=='predict':
        if do_aux is False:
            return reg.predict(inPUT)
        elif do_aux is True:
            return reg.predict(aux_inPUT)
    elif outtype=='aux_predict':
        if do_aux is False:
            return reg.predict(inPUT)-reg.intercept_
        elif do_aux is True:
            return reg.predict(aux_inPUT)-reg.intercept_        
    elif outtype=='coeff':
        return reg.coef_
    
def delete_padding(inTS=None,outTS=None):
    output_nozero,input_nozero = [],[]
    for i in range(len(outTS[:,0])):
        temp = outTS[i,:]
        tempin = inTS[i,:]
        if temp.all()==0:
            continue
        else:
            output_nozero.append(temp)
            input_nozero.append(tempin)
    return input_nozero,output_nozero

def output_regscore(inTS=None,outTS=None,LTlist=None,algorithm='linear',aux_inTS=None,do_aux=False,simplelassoalpha=0.001):
    if do_aux is False:
        return [produce_regscore(delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1],None,'score',False,algorithm,\
                                 dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplelassoalpha) for i in range(len(outTS))]
    elif do_aux is True:
        return [produce_regscore(delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1],aux_inTS[:-int(LTlist[i])],'score',True,algorithm,\
                                 dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplelassoalpha) for i in range(len(outTS))]
    
def output_regscore_withmemory(inTS=None,outTS=None,LTlist=None,algorithm='linear',aux_inTS=None,do_aux=False,simplelassoalpha=0.001):
    if do_aux is False:
        output = []
        for i in (range(len(outTS))):
            inTSn,outTSn = delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1]
            inTRAIN = np.concatenate(((inTSn[int(LTlist[i]):]),(inTSn[0:-int(LTlist[i])])),axis=1)
            outTRAIN = outTSn[int(LTlist[i]):]
            output.append(produce_regscore(inTRAIN,outTRAIN,None,'score',False,algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplelassoalpha))
        return output
    elif do_aux is True:
        output = []
        for i in (range(len(outTS))):
            inTSn,outTSn = delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1]
            inTRAIN = np.concatenate(((inTSn[int(LTlist[i]):]),(inTSn[0:-int(LTlist[i])])),axis=1)
            outTRAIN = outTSn[int(LTlist[i])]
            auxin = np.concatenate(((aux_inTS[:-int(LTlist[i])][int(LTlist[i]):]),(aux_inTS[:-int(LTlist[i])][0:-int(LTlist[i])])),axis=1)
            output.append(produce_regscore(inTRAIN,outTRAIN,auxin,'score',True,algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplelassoalpha))
        return output        
    
def output_regcoeff(inTS=None,outTS=None,LTlist=None,algorithm='linear',memory='with',simplelassoalpha=0.001):
    if memory=='with':
        output = []
        for i in range(len(outTS)):
            inarray = np.concatenate(((inTS[:-int(LTlist[i])][int(LTlist[i]):]),(inTS[:-int(LTlist[i])][0:-int(LTlist[i])])),axis=1)
            outarray = outTS[i][int(LTlist[i]):-int(LTlist[i])]
            output.append(produce_regscore(inarray,outarray,'coeff',algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01))
        return output
    elif memory=='without':
        return [produce_regscore(inTS[:-int(LTlist[i])],outTS[i][:-int(LTlist[i])],'coeff',algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplelassoalpha) for i in range(len(outTS))]

class preproc_data:
    def __init__(self,PCAdict=None,folderpath=None,ts_varname=None,nummem=None):
        self.PCAdict = PCAdict
        self.expname=['ctl','ncrf_36h','ncrf_60h','lwcrf']
        self.nummem = nummem # u: 36 (40% variability in du), v:16/32 (40% dv var;50%), w:44 (40% dw var)
        self.ts_varname = ts_varname
    
    def readvar(self,listdict=None,varname=['u','v','w','qv','heatsum'],withtheta='Yes',thetaflat=None,smooth24=False):
        vardict = {}
        if smooth24:
            withtheta='No'
        else:
            withtheta='Yes'
            
        for indx,obj in enumerate(self.expname):
            if withtheta=='Yes':
                templist = [listdict[indx][strvar][24:120] for strvar in varname]
                theta = thetaflat[list(thetaflat.keys())[indx]][24:120]
                templist.insert(3,theta)
                #assert len(templist)==4
                vardict[obj] = templist
            else:
                vardict[obj] = [listdict[indx][strvar][24:120] for strvar in varname]
        return vardict
    
class SimpleIOhandler(BaseEstimator, TransformerMixin):
    def __init__(self,LT=None,auxIN=False):
        self.LT = LT
        self.auxIN = auxIN
    def transform(self,X=None,y=None):
        if self.auxIN is not None:
            Xtrain = delete_padding(self.auxIN,y[int(self.LT)-1])[0]
            return Xtrain
        else:
            Xtrain,ytrain = delete_padding(X,y[int(self.LT)-1])[0],delete_padding(X,y[int(self.LT)-1])[1]
            return Xtrain,ytrain

class model_fitpredict:
    def __init__(self,mlrIN=None,mlrOUT=None,model=None,LT=None):
        self.mlrIN=mlrIN
        self.mlrOUT=mlrOUT
        self.model=model
        self.LT=LT
        
    def modelfit(self,singleLT=False):
        if singleLT:
            try:
                reg = self.model().fit(self.mlrIN,self.mlrOUT)
            except:
                reg = self.model.fit(self.mlrIN,self.mlrOUT)
            return reg,self.mlrIN
        else:
            models,mlrINN=[],[]
            for indx,LTT in (enumerate(self.LT)):
                INt,OUTt = SimpleIOhandler(LT=LTT,auxIN=None).transform(self.mlrIN,self.mlrOUT)
                reg = self.model().fit(INt,OUTt)
                models.append(reg)
                mlrINN.append(INt)
            return models,mlrINN
    
    def modelforecast(self,auxIN=None):
        if auxIN is not None:
            mlrmodels,mlrINPUT = self.modelfit()
            auxINN = [SimpleIOhandler(LT=LTT,auxIN=auxIN).transform(self.mlrIN,self.mlrOUT) for LTT in self.LT]
            forecastPC = [modelz.predict(auxINNz) for modelz,auxINNz in zip(mlrmodels,auxINN)]
        else:
            mlrmodels,mlrINPUT = self.modelfit()
            forecastPC = [modelz.predict(INPUTz) for modelz,INPUTz in zip(mlrmodels,mlrINPUT)]
        return forecastPC

def recon_from_linear(forecast_eiginput=None,PCA_dict=None,LT=None,numcomp=[11,12,12],large_out=False,u=None,v=None,w=None,savepath=None):
		def output_reshapeRECON(forecast_eig=None,PCAdict=None):
				testrec_dudt = np.dot(forecast_eig[:,0:numcomp[0]],(PCA_dict['u'].components_[0:numcomp[0]]))#.reshape((91,39,360,167))
				testrec_dvdt = np.dot(forecast_eig[:,numcomp[0]:numcomp[0]+numcomp[1]],(PCA_dict['v'].components_[0:numcomp[1]]))#.reshape((91,39,360,167))
				testrec_dwdt = np.dot(forecast_eig[:,numcomp[0]+numcomp[1]:],(PCA_dict['w'].components_[0:numcomp[2]]))#.reshape((39,360,167))
				return testrec_dudt,testrec_dvdt,testrec_dwdt
		######################################################################################################################################################
		name = ['dudt','dvdt','dwdt']
		temp1,temp2,temp3 = [],[],[]
		temp1b,temp2b,temp3b = [],[],[]
		for i,num in tqdm(enumerate((LT))):
				teMP1,teMP2,teMP3 = output_reshapeRECON(forecast_eiginput[int(num)-1],PCA_dict)
				reteMP1,reteMP2,reteMP3 = derive_var.windrates_real(u=u,v=v,w=w,LT=int(num))
				# Square error
				if large_out is False:
						temp1.append(np.sum((teMP1-reteMP1)**2))
				else:
						read_and_proc.save_to_pickle(loc=savepath+'full_U_LT'+str(num),var=teMP1)
						read_and_proc.save_to_pickle(loc=savepath+'fullse_U_LT'+str(num),var=teMP1-reteMP1)
				del reteMP1
				gc.collect()

				if large_out is False:
						temp2.append(np.sum((teMP2-reteMP2)**2))
				else:
						read_and_proc.save_to_pickle(loc=savepath+'full_V_LT'+str(num),var=teMP2)
						read_and_proc.save_to_pickle(loc=savepath+'fullse_V_LT'+str(num),var=teMP2-reteMP2)
				del reteMP2
				gc.collect()

				if large_out is False:
						temp3.append(np.sum((teMP3-reteMP3)**2))
				else:
						read_and_proc.save_to_pickle(loc=savepath+'full_W_LT'+str(num),var=teMP3)
						read_and_proc.save_to_pickle(loc=savepath+'fullse_W_LT'+str(num),var=teMP3-reteMP3)
				del reteMP3
				gc.collect()


				# Variance
				if large_out is False:
						temp1b.append(np.sum((teMP1-np.nanmean(teMP1))**2))
				else:
						continue
				del teMP1
				gc.collect()
				if large_out is False:
						temp2b.append(np.sum((teMP2-np.nanmean(teMP2))**2))
				else:
						continue
				del teMP2
				gc.collect()
				if large_out is False:
						temp3b.append(np.sum((teMP3-np.nanmean(teMP3))**2))
				else:
						continue
				del teMP3
				gc.collect()
		del i
				
		if large_out is False:
				se_store = {name[0]:temp1,name[1]:temp2,name[2]:temp3}
				va_store = {name[0]:temp1b,name[1]:temp2b,name[2]:temp3b}
				######################################################################################################################################################
				return se_store,va_store
		else:
				return None
