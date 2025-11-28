import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import xarray as xr
import numpy as np
import glob,os,sys
from tqdm.auto import tqdm
import json,pickle
import dask.array as da
import gc
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc,plotting
from tools.mlr import mlr,proc_mlrfcst,maria_IO,cartesian_retrieve
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan

class preproc_r2:
    def __init__(self,uvwth=None,validindex=None,testindex=None):
        self.uvwth=uvwth
        self.validindex=validindex
        self.testindex=testindex
    
    def forward_diff(self,arrayin=None,delta=None,axis=None,LT=1):
        result = []
        if axis==0:
            for i in range(0,arrayin.shape[axis]-LT):
                temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
                result.append(temp)
            return np.asarray(result)
        
    def get_time_diff_terms_intermediate(self,inputvar=None,LT=None,wantvar=None,case='Haiyan'):
        def _get_time_diff(array=None,timedelta=60*60,LT=None):
            store = []
            for exp in array:
                if case=='Haiyan':
                    try:
                        a = self.forward_diff(np.nanmean(exp.reshape(exp.shape[0],10,360,208),axis=2).reshape(exp.shape[0],-1),timedelta,0,LT)
                    except:
                        a = self.forward_diff(exp,timedelta,0,LT)                      
                elif case=='Maria2D':
                    a = self.forward_diff(exp,timedelta,0,LT)
                if a.shape[0]>0:
                    azero = np.zeros((LT,exp.shape[-1]))
                    store.append(np.asarray(a))
                    #store.append(np.concatenate((a,azero),axis=0))
                else:
                    continue
                    #store.append(np.zeros((exp.shape[0],exp.shape[-1])))
            return store
        
        storedict = {}
        for wantvarZ,wantvarN in enumerate(wantvar):
            storedict[wantvarN] = _get_time_diff(array=inputvar[wantvarZ],LT=LT)
        return storedict
    
    def get_time_diff_terms(self,inputvar=None,LT=None,wantvar=None,case='Haiyan'):
        def _get_time_diff(array=None,timedelta=60*60,LT=None):
            store = []
            for exp in array:
                a = self.forward_diff(exp,timedelta,0,LT)
                if a.shape[0]>0:
                    azero = np.zeros((LT,exp.shape[-1]))
                    store.append(np.asarray(a))
                    #store.append(np.concatenate((a,azero),axis=0))
                else:
                    continue
                    #store.append(np.zeros((exp.shape[0],exp.shape[-1])))
            return store
        
        storedict = {}
        for wantvarZ,wantvarN in enumerate(wantvar):
            storedict[wantvarN] = _get_time_diff(array=inputvar[wantvarZ],LT=LT)
        return storedict
    
    def windrates_real(self,uvwheat=None,LT=None,category='train',withW=False,twoDthreeD='2D' or '3D'):
        if category=='train':
            popindex = self.validindex+self.testindex
            ut = [uvwheat[0][i] for i in range(len(uvwheat[0])) if i not in popindex]
            vt = [uvwheat[1][i] for i in range(len(uvwheat[1])) if i not in popindex]
            if withW is True:
                wt = [uvwheat[2][i] for i in range(len(uvwheat[2])) if i not in popindex]
                thetat = [uvwheat[3][i] for i in range(len(uvwheat[3])) if i not in popindex]
                assert len(ut)==16, 'wrong train-valid-test separation!'
                if twoDthreeD=='2D':
                    dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'],case='Haiyan')
                elif twoDthreeD=='3D':
                    dtermsT = self.get_time_diff_terms(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'],case='Haiyan')                    
            else:
                thetat = [uvwheat[2][i] for i in range(len(uvwheat[2])) if i not in popindex]                
                assert len(ut)==16, 'wrong train-valid-test separation!'
                if twoDthreeD=='2D':
                    dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,thetat],LT=LT,wantvar=['u','v','theta'],case='Haiyan')
                elif twoDthreeD=='3D':
                    dtermsT = self.get_time_diff_terms(inputvar=[ut,vt,thetat],LT=LT,wantvar=['u','v','theta'],case='Haiyan')
        elif category=='valid':
            uv = [uvwheat[0][index] for index in self.validindex]
            vv = [uvwheat[1][index] for index in self.validindex]
            if withW is True:
                wv = [uvwheat[2][index] for index in self.validindex]
                thetav = [uvwheat[3][index] for index in self.validindex]
                if twoDthreeD=='2D':
                    dtermsT = self.get_time_diff_terms_intermediate(inputvar=[uv,vv,wv,thetav],LT=LT,wantvar=['u','v','w','theta'],case='Haiyan')
                elif twoDthreeD=='3D':
                    dtermsT = self.get_time_diff_terms(inputvar=[uv,vv,wv,thetav],LT=LT,wantvar=['u','v','w','theta'],case='Haiyan')
            else:
                thetav = [uvwheat[2][index] for index in self.validindex]
                if twoDthreeD=='2D':
                    dtermsT = self.get_time_diff_terms_intermediate(inputvar=[uv,vv,thetav],LT=LT,wantvar=['u','v','theta'],case='Haiyan')
                elif twoDthreeD=='3D':
                    dtermsT = self.get_time_diff_terms(inputvar=[uv,vv,thetav],LT=LT,wantvar=['u','v','theta'],case='Haiyan')
        elif category=='test':
            ut = [uvwheat[0][index] for index in self.testindex]
            vt = [uvwheat[1][index] for index in self.testindex]
            if withW is True:
                wt = [uvwheat[2][index] for index in self.testindex]
                thetat = [uvwheat[3][index] for index in self.testindex]
                if twoDthreeD=='2D':
                    dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'],case='Haiyan')
                elif twoDthreeD=='3D':
                    dtermsT = self.get_time_diff_terms(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'],case='Haiyan')
            else:
                thetat = [uvwheat[2][index] for index in self.testindex]
                if twoDthreeD=='2D':
                    dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,thetat],LT=LT,wantvar=['u','v','theta'],case='Haiyan')
                elif twoDthreeD=='3D':
                    dtermsT = self.get_time_diff_terms(inputvar=[ut,vt,thetat],LT=LT,wantvar=['u','v','theta'],case='Haiyan')                    
        
        dudt = np.concatenate([testx for testx in dtermsT['u']],axis=0)
        dvdt = np.concatenate([testx for testx in dtermsT['v']],axis=0)
        if withW is True:
            dwdt = np.concatenate([testx for testx in dtermsT['w']],axis=0)
            dthdt = np.concatenate([testx for testx in dtermsT['theta']],axis=0)
            index = [testx.shape for testx in dtermsT['theta']]
            dictt = {'du':dudt,'dv':dvdt,'dw':dwdt,'dth':dthdt,'index':index}
            del dtermsT,dudt,dvdt,dwdt,dthdt,index
            gc.collect()
        else:
            dthdt = np.concatenate([testx for testx in dtermsT['theta']],axis=0)
            index = [testx.shape for testx in dtermsT['theta']]
            dictt = {'du':dudt,'dv':dvdt,'dth':dthdt,'index':index}
            del dtermsT,dudt,dvdt,dthdt,index
            gc.collect()            
        return dictt
    
    def windrates_real_maria(self,uvwheat=None,LT=None,category='train',testindex=[2,12],withW=False):        
        if category=='train':
            ut = [uvwheat[0][i] for i in range(len(uvwheat[0])) if i not in testindex]
            vt = [uvwheat[1][i] for i in range(len(uvwheat[1])) if i not in testindex]
            if withW is True:
                wt = [uvwheat[2][i] for i in range(len(uvwheat[2])) if i not in testindex]
                thetat = [uvwheat[3][i] for i in range(len(uvwheat[3])) if i not in testindex]
                assert len(ut)==4, 'wrong train-valid-test separation!'
                dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'],case='Maria2D')
            else:
                thetat = [uvwheat[2][i] for i in range(len(uvwheat[2])) if i not in testindex]
                assert len(ut)==4, 'wrong train-valid-test separation!'
                dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,thetat],LT=LT,wantvar=['u','v','theta'],case='Maria2D')                
        elif category=='test':
            ut = [uvwheat[0][index] for index in testindex]
            vt = [uvwheat[1][index] for index in testindex]
            if withW is True:
                wt = [uvwheat[2][index] for index in testindex]
                thetat = [uvwheat[3][index] for index in testindex]
                dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'],case='Maria2D')
            else:
                thetat = [uvwheat[2][index] for index in testindex]
                dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,thetat],LT=LT,wantvar=['u','v','theta'],case='Maria2D')                
        
        dudt = np.concatenate([testx for testx in dtermsT['u']],axis=0)
        dvdt = np.concatenate([testx for testx in dtermsT['v']],axis=0)
        if withW is True:
            dwdt = np.concatenate([testx for testx in dtermsT['w']],axis=0)
            dthdt = np.concatenate([testx for testx in dtermsT['theta']],axis=0)
            index = [testx.shape for testx in dtermsT['theta']]
            dictt = {'du':dudt,'dv':dvdt,'dw':dwdt,'dth':dthdt,'index':index}
            del dtermsT,dudt,dvdt,dwdt,dthdt,index
        else:
            dthdt = np.concatenate([testx for testx in dtermsT['theta']],axis=0)
            index = [testx.shape for testx in dtermsT['theta']]
            dictt = {'du':dudt,'dv':dvdt,'dth':dthdt,'index':index}
            del dtermsT,dudt,dvdt,dthdt,index            
        gc.collect()
        return dictt
    
    def output_reshapeRECON(self,forecast_eig=None,PCA_dict=None,numcomp=None,withW=True):
        testrec_dudt = np.dot(forecast_eig[:,0:numcomp[0]],(PCA_dict['u'].components_[0:numcomp[0]]))#.reshape((91,39,360,167))
        testrec_dvdt = np.dot(forecast_eig[:,numcomp[0]:numcomp[0]+numcomp[1]],(PCA_dict['v'].components_[0:numcomp[1]]))#.reshape((91,39,360,167))
        if withW is True:
            testrec_dwdt = np.dot(forecast_eig[:,numcomp[0]+numcomp[1]:numcomp[0]+numcomp[1]+numcomp[2]],(PCA_dict['w'].components_[0:numcomp[2]]))#.reshape((39,360,167))
            testrec_dthdt = np.dot(forecast_eig[:,numcomp[0]+numcomp[1]+numcomp[2]:],(PCA_dict['theta'].components_[0:numcomp[3]]))#.reshape((39,360,167))
            return testrec_dudt,testrec_dvdt,testrec_dwdt,testrec_dthdt
        else:
            testrec_dthdt = np.dot(forecast_eig[:,numcomp[0]+numcomp[1]:numcomp[0]+numcomp[1]+numcomp[2]],(PCA_dict['theta'].components_[0:numcomp[2]]))#.reshape((39,360,167))
            return testrec_dudt,testrec_dvdt,testrec_dthdt
        
    def conversion_predictPC(self,yforecast=None,mshpe=[39,360,167],PCA_dict=None,numcomp=None,withW=True):
        if withW is True:
            t1,t2,t3,t4 = self.output_reshapeRECON(yforecast,PCA_dict,numcomp,withW)
            return t1,t2,t3,t4
        else:
            t1,t2,t3 = self.output_reshapeRECON(yforecast,PCA_dict,numcomp,withW)
            return t1,t2,t3            
        
    def _back_to_exp(self,timeseries=None,divider=None):
        printout = [timeseries[0:divider[0],:]]
        for i in range(1,len(divider)-1):
            printout.append(timeseries[divider[i-1]:divider[i],:])
        printout.append(timeseries[divider[-2]:,:])
        return printout
    
    def back_to_exp(self,inputlong=None,divider=None,senvarname=None):
        ts_dict = {}
        if senvarname is None:
            for indx,obj in tqdm(enumerate(self.varname)):
                ts_dict[obj] = self._back_to_exp(inputlong[obj],divider)
        else:
            for indx,obj in tqdm(enumerate(senvarname)):
                ts_dict[obj] = self._back_to_exp(inputlong[obj],divider)            
        return ts_dict
    
class _get_r2:
    def __init__(self,pcadict=None,afdict=None,numcomp=None,LT=None,FFWmodel=None,reducedX=None,realVARS=None,case='Haiyan_axisym',suffix=None,single=True):
        self.pcadict=pcadict
        self.afdict=afdict
        self.numcomp=numcomp
        self.lti=LT
        self.FFWmodel=FFWmodel
        self.reducedX=reducedX
        self.realvar = realVARS
        self.case = case
        self.suffix=suffix

    def _get_trainR2(self):
        return [cartesian_retrieve.retrieve_cartesian(PCA_dict=self.pcadict,Af_dict=self.afdict,numcomp=self.numcomp,LT=self.lti,forecastPC=None,target='all',\
                              suffix=self.suffix).output_r2(FFWmodels=self.FFWmodel[i],reducedX=self.reducedX[i],realU=self.realvar[i]['du'],realV=self.realvar[i]['dv'],realW=self.realvar[i]['dw'],realTH=self.realvar[i]['dth'],case=self.case)
               for i in tqdm(range(len(self.FFWmodel)))]
    
    def _get_validR2(self,X=None,y=None,locselected=[121,122,123,124],newfeature=None,selectlocloop=False):
        def _get_loopfeature(X=None,y=None,timeindx=None,lochold=None):
            if self.case=='Haiyan_axisym':
                yvalid = [yobj[1] for yobj in y[timeindx]]
                mlrIN_v,mlrOUT_v = mlr.SimpleIOhandler(LT=self.lti,auxIN=None).transform(X[timeindx]['dtthuvwqv'],yvalid)
            elif self.case=='Maria_axisym':
                try:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[1][timeindx])
                except:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[timeindx])                    
            loopfeature = [np.asarray(lochold.copy())]#[np.asarray(locselected.copy())]
            for i in range(len(newfeature[timeindx])):
                loopfeature.append(np.append(loopfeature[i],newfeature[timeindx][i]))
            return loopfeature,mlrIN_v,mlrOUT_v
        r2store = []
        for i in range(len(self.FFWmodel)):
            if selectlocloop is False:
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,locselected)
            else:
                loctoin = [locselected[i]]
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,loctoin)
                print((loopfeature))
            reducedX_test = [np.asarray(mlrIN_v)[:,sorted(loopfeature[i])] for i in range(len(loopfeature))]
            r2_train = cartesian_retrieve.retrieve_cartesian(PCA_dict=self.pcadict,Af_dict=self.afdict,numcomp=self.numcomp,LT=self.lti,forecastPC=None,target='all',\
                                          suffix=self.suffix).output_r2(FFWmodels=self.FFWmodel[i],reducedX=reducedX_test[1:],realU=self.realvar[i]['du'],realV=self.realvar[i]['dv'],realW=self.realvar[i]['dw'],realTH=self.realvar[i]['dth'],
                                                                   case=self.case)
            r2store.append(r2_train)
        return r2store
            
    def _get_testR2(self,X=None,y=None,locselected=[121,122,123,124],newfeature=None):
        def _get_loopfeature(X=None,y=None,timeindx=None):
            yvalid = [yobj[2] for yobj in y[timeindx]]
            mlrIN_v,mlrOUT_v = mlr.SimpleIOhandler(LT=self.lti,auxIN=None).transform(X[timeindx]['dtthuvwqv'],yvalid)
                
            loopfeature = [np.asarray(locselected.copy())]
            for i in range(len(newfeature[timeindx])):
                loopfeature.append(np.append(loopfeature[i],newfeature[timeindx][i]))
            return loopfeature,mlrIN_v,mlrOUT_v
        r2store = []
        for i in tqdm(range(len(self.FFWmodel))):
            loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i)
            reducedX_test = [np.asarray(mlrIN_v)[:,sorted(loopfeature[i])] for i in range(len(loopfeature))]
            r2_train = cartesian_retrieve.retrieve_cartesian(PCA_dict=self.pcadict,Af_dict=self.afdict,numcomp=self.numcomp,LT=self.lti,forecastPC=None,target='all',\
                                          suffix=self.suffix).output_r2(FFWmodels=self.FFWmodel[i],reducedX=reducedX_test[1:],realU=self.realvar[i]['du'],realV=self.realvar[i]['dv'],realW=self.realvar[i]['dw'],realTH=self.realvar[i]['dth'],
                                                                   case=self.case)
            r2store.append(r2_train)
        return r2store

    def _get_trainX(self,X=None,y=None,locselected=[121,122,123,124],newfeature=None,selectlocloop=False):
        def _get_loopfeature(X=None,y=None,timeindx=None,lochold=None):
            if self.case=='Haiyan_axisym':
                yvalid = [yobj[0] for yobj in y[timeindx]]
                mlrIN_v,mlrOUT_v = mlr.SimpleIOhandler(LT=self.lti,auxIN=None).transform(X[timeindx]['dtthuvwqv'],yvalid)
            elif self.case=='Maria_axisym':
                try:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[0][timeindx])
                except:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[timeindx])                    
            loopfeature = [np.asarray(lochold.copy())]#[np.asarray(locselected.copy())]
            for i in range(len(newfeature[timeindx])):
                loopfeature.append(np.append(loopfeature[i],newfeature[timeindx][i]))
            return loopfeature,mlrIN_v,mlrOUT_v
        store = []
        for i in range(len(self.FFWmodel)):
            if selectlocloop is False:
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,locselected)
            else:
                loctoin = [locselected[i]]
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,loctoin)
            reducedX_test = [np.asarray(mlrIN_v)[:,sorted(loopfeature[i])] for i in range(len(loopfeature))]
            store.append(reducedX_test)
        return store
    
    def _get_validX(self,X=None,y=None,locselected=[121,122,123,124],newfeature=None,selectlocloop=False):
        def _get_loopfeature(X=None,y=None,timeindx=None,lochold=None):
            if self.case=='Haiyan_axisym':
                yvalid = [yobj[1] for yobj in y[timeindx]]
                mlrIN_v,mlrOUT_v = mlr.SimpleIOhandler(LT=self.lti,auxIN=None).transform(X[timeindx]['dtthuvwqv'],yvalid)
            elif self.case=='Maria_axisym':
                try:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[1][timeindx])
                except:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[timeindx])                    
            loopfeature = [np.asarray(lochold.copy())]#[np.asarray(locselected.copy())]
            for i in range(len(newfeature[timeindx])):
                loopfeature.append(np.append(loopfeature[i],newfeature[timeindx][i]))
            return loopfeature,mlrIN_v,mlrOUT_v
        store = []
        for i in range(len(self.FFWmodel)):
            if selectlocloop is False:
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,locselected)
            else:
                loctoin = [locselected[i]]
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,loctoin)
            reducedX_test = [np.asarray(mlrIN_v)[:,sorted(loopfeature[i])] for i in range(len(loopfeature))]
            store.append(reducedX_test)
        return store
    
            
    def _get_testX(self,X=None,y=None,locselected=[121,122,123,124],newfeature=None,selectlocloop=False):
        def _get_loopfeature(X=None,y=None,timeindx=None,lochold=None):
            if self.case=='Haiyan_axisym':
                yvalid = [yobj[2] for yobj in y[timeindx]]
                mlrIN_v,mlrOUT_v = mlr.SimpleIOhandler(LT=self.lti,auxIN=None).transform(X[timeindx]['dtthuvwqv'],yvalid)
            elif self.case=='Maria_axisym':
                try:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[2][timeindx])
                except:
                    mlrIN_v,mlrOUT_v=mlr.delete_padding(X[timeindx],y[timeindx])                    
            loopfeature = [np.asarray(lochold.copy())]#[np.asarray(locselected.copy())]
            for i in range(len(newfeature[timeindx])):
                loopfeature.append(np.append(loopfeature[i],newfeature[timeindx][i]))
            return loopfeature,mlrIN_v,mlrOUT_v
        store = []
        for i in tqdm(range(len(self.FFWmodel))):
            if selectlocloop is False:
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,locselected)
            else:
                loctoin = [locselected[i]]
                loopfeature,mlrIN_v,_ = _get_loopfeature(X,y,i,loctoin)
            reducedX_test = [np.asarray(mlrIN_v)[:,sorted(loopfeature[i])] for i in range(len(loopfeature))]
            store.append(reducedX_test)
        return store