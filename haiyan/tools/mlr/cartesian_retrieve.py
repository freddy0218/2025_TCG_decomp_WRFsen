import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import cross_val_score
from tools import derive_var,read_and_proc,preproc_noensemble
from tools.mlr import mlr,proc_mlrfcst,maria_IO,ffs
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
import gc

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
        return np.asarray(result)
    
class retrieve_cartesian:
    def __init__(self,path=None,PCA_dict=None,Af_dict=None,numcomp=[11,11,15],LT=None,forecastPC=None,target='all',suffix=None,case='Maria'):
        self.PCA_dict=PCA_dict
        self.numcomp=numcomp
        self.forecastPC = forecastPC
        self.LT = LT
        self.target=target
        self.suffix=suffix
        self.case=case
        self.path=path
        
    def get_time_diff_terms_intermediate(self,inputvar=None,LT=None,wantvar=None):
        def _get_time_diff(array=None,timedelta=60*60,LT=None):
            if self.case=='Maria':
                sizesets = [39,360,int(array[0].shape[1]/39/360)]
            elif self.case=='Haiyan':
                sizesets = [10,360,int(array[0].shape[1]/10/360)]
            store = []
            for exp in array:
                a = forward_diff(np.nanmean(exp.reshape(exp.shape[0],sizesets[0],sizesets[1],sizesets[2]),axis=2).reshape(exp.shape[0],-1),timedelta,0,LT)
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
    
    def windrates_real_maria(self,uvwheatpath=None,LT=None,category='train',testindex=[2,12]):
        u = [read_and_proc.depickle(self.path+'TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/preproc2/'+str(lime)+self.suffix)['u'] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
        v = [read_and_proc.depickle(self.path+'TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/preproc2/'+str(lime)+self.suffix)['v'] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
        w = [read_and_proc.depickle(self.path+'TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/preproc2/'+str(lime)+self.suffix)['w'] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
        theta = [read_and_proc.depickle(self.path+'TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/preproc2/'+str(lime)+self.suffix)['theta'] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
        
        if category=='train':
            ut = [u[i] for i in range(len(u)) if i not in testindex]
            vt = [v[i] for i in range(len(v)) if i not in testindex]
            wt = [w[i] for i in range(len(w)) if i not in testindex]
            thetat = [theta[i] for i in range(len(theta)) if i not in testindex]
            assert len(ut)==4, 'wrong train-valid-test separation!'
            dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'])
        elif category=='test':
            ut = [u[index] for index in testindex]
            vt = [v[index] for index in testindex]
            wt = [w[index] for index in testindex]
            thetat = [theta[index] for index in testindex]            
            del u,v,w,theta
            gc.collect()
            dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'])
        
        tempdict = {}
        for obj in ['u','v','w','theta']:
            tempdict[obj] = dtermsT[obj]
        del dtermsT
        gc.collect()
        return tempdict
    
    def windrates_real_haiyan(self,uvwheatpath=None,LT=None,category='train',validindex=[1,6],testindex=[2,12]):
        u = [read_and_proc.depickle(self.path+uvwheatpath+'mem'+str(lime)+self.suffix)['u'] for lime in (range(1,21))].copy()
        v = [read_and_proc.depickle(self.path+uvwheatpath+'mem'+str(lime)+self.suffix)['v'] for lime in (range(1,21))].copy()
        w = [read_and_proc.depickle(self.path+uvwheatpath+'mem'+str(lime)+self.suffix)['w'] for lime in (range(1,21))].copy()
        theta = [read_and_proc.depickle(self.path+uvwheatpath+'mem'+str(lime)+self.suffix)['theta'] for lime in (range(1,21))].copy()
        
        if category=='train':
            popindex = validindex+testindex
            ut = [u[i] for i in range(len(u)) if i not in popindex]
            vt = [v[i] for i in range(len(v)) if i not in popindex]
            wt = [w[i] for i in range(len(w)) if i not in popindex]
            thetat = [theta[i] for i in range(len(theta)) if i not in popindex]
            assert len(ut)==16, 'wrong train-valid-test separation!'
            dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'])
        elif category=='valid':
            uv = [u[index] for index in validindex]
            vv = [v[index] for index in validindex]
            wv = [w[index] for index in validindex]
            thetav = [theta[index] for index in validindex]
            del u,v,w,theta
            gc.collect()
            dtermsT = self.get_time_diff_terms_intermediate(inputvar=[uv,vv,wv,thetav],LT=LT,wantvar=['u','v','w','theta'])
        elif category=='test':
            ut = [u[index] for index in testindex]
            vt = [v[index] for index in testindex]
            wt = [w[index] for index in testindex]
            thetat = [theta[index] for index in testindex]            
            del u,v,w,theta
            gc.collect()
            dtermsT = self.get_time_diff_terms_intermediate(inputvar=[ut,vt,wt,thetat],LT=LT,wantvar=['u','v','w','theta'])            
        
        dudt = np.concatenate([testx for testx in dtermsT['u']],axis=0)
        dvdt = np.concatenate([testx for testx in dtermsT['v']],axis=0)
        dwdt = np.concatenate([testx for testx in dtermsT['w']],axis=0)
        dthdt = np.concatenate([testx for testx in dtermsT['theta']],axis=0)
        del dtermsT
        gc.collect()
        return dudt,dvdt,dwdt,dthdt
    
    
    def output_reshapeRECON(self,forecast_eig=None):
        if (self.target=='surface') or (self.target=='alluv'):
            testrec_dudt = np.dot(forecast_eig[:,0:self.numcomp[0]],(self.PCA_dict['u'].components_[0:self.numcomp[0]]))#.reshape((91,39,360,167))
            testrec_dvdt = np.dot(forecast_eig[:,self.numcomp[0]:self.numcomp[0]+self.numcomp[1]],(self.PCA_dict['v'].components_[0:self.numcomp[1]]))#.reshape((91,39,360,167))
            return testrec_dudt,testrec_dvdt
        else:
            testrec_dudt = np.dot(forecast_eig[:,0:self.numcomp[0]],(self.PCA_dict['u'].components_[0:self.numcomp[0]]))#.reshape((91,39,360,167))
            testrec_dvdt = np.dot(forecast_eig[:,self.numcomp[0]:self.numcomp[0]+self.numcomp[1]],(self.PCA_dict['v'].components_[0:self.numcomp[1]]))#.reshape((91,39,360,167))
            testrec_dwdt = np.dot(forecast_eig[:,self.numcomp[0]+self.numcomp[1]:self.numcomp[0]+self.numcomp[1]+self.numcomp[2]],(self.PCA_dict['w'].components_[0:self.numcomp[2]]))#.reshape((39,360,167))
            testrec_dthdt = np.dot(forecast_eig[:,self.numcomp[0]+self.numcomp[1]+self.numcomp[2]:],(self.PCA_dict['theta'].components_[0:self.numcomp[3]]))#.reshape((39,360,167))
            return testrec_dudt,testrec_dvdt,testrec_dwdt,testrec_dthdt
        
    def conversion_predictPC(self,yforecast=None,mshpe=[39,360,167]):
        if self.target=='surface':
            t1,t2 = self.output_reshapeRECON(forecast_eig=yforecast)
            return (t1.reshape(t1.shape[0],mshpe[0],mshpe[1],mshpe[2])[:,0,:,:]).reshape(t1.shape[0],mshpe[1]*mshpe[2]),(t2.reshape(t2.shape[0],mshpe[0],mshpe[1],mshpe[2])[:,0,:,:]).reshape(t2.shape[0],mshpe[1]*mshpe[2])
        elif self.target=='all': 
            t1,t2,t3,t4 = self.output_reshapeRECON(forecast_eig=yforecast)
            return t1,t2,t3,t4
        
    def output_r2(self,FFWmodels=None,reducedX=None,realU=None,realV=None,realW=None,realTH=None,case='Haiyan'):
        yf = [models.predict(Xns) for (models,Xns) in zip(FFWmodels[:],reducedX[:])]
        mlr_r2 = []
        if case=='Maria_axisym': num=30
        elif case=='1step_other': num=45
        else: num=len(yf)
        
        for i in (range(len(yf))):#len(yf))):
            if case=='Maria':
                temp1,temp2 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,360,167])
                mlr_r2.append(r2_score(np.concatenate((realU,realV),axis=0),np.concatenate((temp1,temp2),axis=0)))
            elif case=='Haiyan':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[10,360,208])
                a_u,a_v,a_w,a_th = r2_score(realU,temp1),r2_score(realV,temp2),r2_score(realW,temp3),r2_score(realTH,temp4)
                a = (a_u+a_v+a_w+a_th)/4#r2_score(np.concatenate((realU,realV,realW,realTH),axis=0),np.concatenate((temp1,temp2,temp3,temp4),axis=0))
                mlr_r2.append(a)
            elif case=='Haiyan_axisym':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[10,208])
                a_u,a_v,a_w,a_th = r2_score(realU,temp1),r2_score(realV,temp2),r2_score(realW,temp3),r2_score(realTH,temp4)
                a = (a_u+a_v+a_w+a_th)/4
                mlr_r2.append(a)
            elif case=='Maria_axisym':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,200])
                a_u,a_v,a_w,a_th = r2_score(realU,temp1),r2_score(realV,temp2),r2_score(realW,temp3),r2_score(realTH,temp4)
                a = (a_u+a_v+a_w+a_th)/4
                mlr_r2.append(a)
            del temp1,temp2
            gc.collect()
        return mlr_r2
    
    def output_prediction(self,FFWmodels=None,reducedX=None,case='Haiyan'):
        yf = [models.predict(Xns) for (models,Xns) in zip(FFWmodels[:],reducedX[:])]
        du,dv,dw,dth = [],[],[],[]
        if case=='Maria_axisym': num=30
        elif case=='1step_other': num=45
        else: num=len(yf)
        
        for i in (range(len(yf))):#len(yf))):
            if case=='Maria':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,360,167])
            elif case=='Haiyan':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[10,360,208])
            elif case=='Haiyan_axisym':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[10,208])
            elif case=='Maria_axisym':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,200])
            du.append(temp1)
            dv.append(temp2)
            dw.append(temp3)
            dth.append(temp4)
            del temp1,temp2,temp3,temp4
            gc.collect()
        return {'du':du,'dv':dv,'dw':dw,'dth':dth}
    
    def output_prediction_decomp(self,FFWmodels=None,reducedX=None,case='Haiyan'):
        yf = [models.predict(Xns)-models.intercept_ for (models,Xns) in zip(FFWmodels[:],reducedX[:])]
        du,dv,dw,dth = [],[],[],[]
        if case=='Maria_axisym': num=30
        elif case=='1step_other': num=45
        else: num=len(yf)
        
        for i in (range(len(yf))):#len(yf))):
            if case=='Maria':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,360,167])
            elif case=='Haiyan':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[10,360,208])
            elif case=='Haiyan_axisym':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[10,208])
            elif case=='Maria_axisym':
                temp1,temp2,temp3,temp4 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,200])
            du.append(temp1)
            dv.append(temp2)
            dw.append(temp3)
            dth.append(temp4)
            del temp1,temp2,temp3,temp4
            gc.collect()
        return {'du':du,'dv':dv,'dw':dw,'dth':dth}