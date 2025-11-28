from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from tools import derive_var,read_and_proc
from tools.preprocess import do_eof,preproc_maria
from tqdm import tqdm
import numpy as np
import gc

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
        return np.asarray(result)
    
class retrieve_cartesian:
    def __init__(self,PCA_dict=None,Af_dict=None,numcomp=[11,11,15],LT=None,forecastPC=None,target='all'):
        self.PCA_dict=PCA_dict
        self.numcomp=numcomp
        self.forecastPC = forecastPC
        self.u = [Af_dict['ctl'][0],Af_dict['ncrf_36h'][0][(36-24-2):],Af_dict['ncrf_60h'][0][(60-24-2):],Af_dict['lwcrf'][0][(36-24-2):]]
        self.v = [Af_dict['ctl'][1],Af_dict['ncrf_36h'][1][(36-24-2):],Af_dict['ncrf_60h'][1][(60-24-2):],Af_dict['lwcrf'][1][(36-24-2):]]
        self.w = [Af_dict['ctl'][2],Af_dict['ncrf_36h'][2][(36-24-2):],Af_dict['ncrf_60h'][2][(60-24-2):],Af_dict['lwcrf'][2][(36-24-2):]]
        self.th = [Af_dict['ctl'][3],Af_dict['ncrf_36h'][3][(36-24-2):],Af_dict['ncrf_60h'][3][(60-24-2):],Af_dict['lwcrf'][3][(36-24-2):]]
        self.LT = LT
        self.target=target
    
    def windrates_real(self,LT=None):
        dudtT = [forward_diff(uobj,60*60,0,LT) for uobj in self.u]
        dudt = np.concatenate((dudtT[0],dudtT[1],dudtT[2],dudtT[3]),axis=0)
        dvdtT = [forward_diff(vobj,60*60,0,LT) for vobj in self.v]
        dvdt = np.concatenate((dvdtT[0],dvdtT[1],dvdtT[2],dvdtT[3]),axis=0)
        dwdtT = [forward_diff(wobj,60*60,0,LT) for wobj in self.w]
        dwdt = np.concatenate((dwdtT[0],dwdtT[1],dwdtT[2],dwdtT[3]),axis=0)
        dthdtT = [forward_diff(thobj,60*60,0,LT) for thobj in self.th]
        dthdt = np.concatenate((dthdtT[0],dthdtT[1],dthdtT[2],dthdtT[3]),axis=0)
        del dudtT,dvdtT,dwdtT,dthdtT
        gc.collect()
        return dudt,dvdt,dwdt,dthdt
    
    def output_reshapeRECON(self,forecast_eig=None):
        if self.target=='surface':
            testrec_dudt = np.dot(forecast_eig[:,0:self.numcomp[0]],(self.PCA_dict['u'].components_[0:self.numcomp[0]]))#.reshape((91,39,360,167))
            testrec_dvdt = np.dot(forecast_eig[:,self.numcomp[0]:self.numcomp[0]+self.numcomp[1]],(self.PCA_dict['v'].components_[0:self.numcomp[1]]))#.reshape((91,39,360,167))
            return testrec_dudt,testrec_dvdt
        else:
            testrec_dudt = np.dot(forecast_eig[:,0:self.numcomp[0]],(self.PCA_dict['u'].components_[0:self.numcomp[0]]))#.reshape((91,39,360,167))
            testrec_dvdt = np.dot(forecast_eig[:,self.numcomp[0]:self.numcomp[0]+self.numcomp[1]],(self.PCA_dict['v'].components_[0:self.numcomp[1]]))#.reshape((91,39,360,167))
            testrec_dwdt = np.dot(forecast_eig[:,self.numcomp[0]+self.numcomp[1]:self.numcomp[0]+self.numcomp[1]+self.numcomp[2]],(self.PCA_dict['w'].components_[0:self.numcomp[2]]))#.reshape((39,360,167))
            testrec_dthdt = np.dot(forecast_eig[:,self.numcomp[0]+self.numcomp[1]+self.numcomp[2]:],(self.PCA_dict['theta'].components_[0:self.numcomp[3]]))#.reshape((39,360,167))
            return testrec_dudt,testrec_dvdt,testrec_dwdt,testrec_dthdt
    
    def recon_R2_from_linear(self,saveloc=None,TEST='test',target='full'):
        name = ['dudt','dvdt','dwdt']#,'dthdt']
        r2_store = []
        def APPEND(list1,list2,var1,var2):
            list1.append(var1)
            list2.append(var2)
            return None
        if TEST=='test':
            length = 2
        else:
            length = len(self.LT)
        for i in (range(length)):
            teMP1,teMP2,teMP3,teMP4 = self.output_reshapeRECON(forecast_eig=self.forecastPC[int(self.LT[i]-1)])
            reteMP1,reteMP2,reteMP3,reteMP4 = self.windrates_real(LT=int(self.LT[i]))
            if target=='full':
                tempr2= r2_score(np.concatenate((reteMP1,reteMP2,reteMP3,reteMP4),axis=0),np.concatenate((teMP1,teMP2,teMP3,teMP4),axis=0))
            elif target=='surface':
                teMP1s,teMP2s = (teMP1.reshape(teMP1.shape[0],39,360,167)[:,0,:,:]).reshape(teMP1.shape[0],360*167),(teMP2.reshape(teMP2.shape[0],39,360,167)[:,0,:,:]).reshape(teMP2.shape[0],360*167)
                reteMP1s,reteMP2s = (reteMP1.reshape(reteMP1.shape[0],39,360,167)[:,0,:,:]).reshape(reteMP1.shape[0],360*167),(reteMP2.reshape(reteMP2.shape[0],39,360,167)[:,0,:,:]).reshape(reteMP2.shape[0],360*167)
                tempr2= r2_score(np.concatenate((reteMP1s,reteMP2s),axis=0),np.concatenate((teMP1s,teMP2s),axis=0))
            #tempr2= r2_score(np.concatenate((reteMP1,reteMP2,reteMP3),axis=0),np.concatenate((teMP1,teMP2,teMP3),axis=0))
            if TEST=='test':
                print(tempr2)
            print(tempr2)
            r2_store.append(tempr2)
            del teMP1,teMP2,teMP3,teMP4,reteMP1,reteMP2,reteMP3,reteMP4
            #del teMP4,reteMP4
            gc.collect()
            
        if saveloc is not None:
            read_and_proc.save_to_pickle(loc=saveloc,var=r2_store,TYPE='PICKLE')
            return r2_store
        else:
            return r2_store
    
    def output_forecast_structure(self,LTchoose=None,expname=None,timestep=None,savepath=None):
        """
        expname: 'ctl','ncrf_36h','ncrf_60h','lwcrf'
        """
        def save_forecast(timestep=None,LT=None,expname=None,savepath=None,teMP=None,setting=None):
            # Setting: [expstart,rsindx,removetimes,sen_exp] = [96,36,1,'Yes']
            expstart,rsindx,removetimes,sen_exp=setting[0],setting[1],setting[2],setting[3]
            if sen_exp=='Yes':
                if LT<10:
                    read_and_proc.save_to_pickle(loc=savepath+'/du_'+str(expname)+'_'+str(int(timestep))+'_0'+str(int(LT)),var=teMP[0][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dv_'+str(expname)+'_'+str(int(timestep))+'_0'+str(int(LT)),var=teMP[1][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dw_'+str(expname)+'_'+str(int(timestep))+'_0'+str(int(LT)),var=teMP[2][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dth_'+str(expname)+'_'+str(int(timestep))+'_0'+str(int(LT)),var=teMP[3][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                else:
                    read_and_proc.save_to_pickle(loc=savepath+'/du_'+str(expname)+'_'+str(int(timestep))+'_'+str(int(LT)),var=teMP[0][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dv_'+str(expname)+'_'+str(int(timestep))+'_'+str(int(LT)),var=teMP[1][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dw_'+str(expname)+'_'+str(int(timestep))+'_'+str(int(LT)),var=teMP[2][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dth_'+str(expname)+'_'+str(int(timestep))+'_'+str(int(LT)),var=teMP[3][(expstart-removetimes*int(LT))+(int(timestep)-rsindx),:],TYPE='PICKLE')                    
            else:
                if LT<10:
                    read_and_proc.save_to_pickle(loc=savepath+'/du_'+str(int(timestep))+'_'+'0'+str(int(LT)),var=teMP1[int(timestep)-24,:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dv_'+str(int(timestep))+'_'+'0'+str(int(LT)),var=teMP2[int(timestep)-24,:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dw_'+str(int(timestep))+'_'+'0'+str(int(LT)),var=teMP3[int(timestep)-24,:],TYPE='PICKLE')
                else:
                    read_and_proc.save_to_pickle(loc=savepath+'/du_'+str(int(timestep))+'_'+str(int(LT)),var=teMP1[int(timestep)-24,:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dv_'+str(int(timestep))+'_'+str(int(LT)),var=teMP2[int(timestep)-24,:],TYPE='PICKLE')
                    read_and_proc.save_to_pickle(loc=savepath+'/dw_'+str(int(timestep))+'_'+str(int(LT)),var=teMP3[int(timestep)-24,:],TYPE='PICKLE')
            return None
                    
        for indx,LTobj in tqdm(enumerate(LTchoose)):
            teMP1,teMP2,teMP3,teMP4 = self.output_reshapeRECON(forecast_eig=self.forecastPC[int(LTobj-1)])
            
            if expname=='ncrf_36h':
                expstart,rsindx,removetimes,sen_exp=96,34,1,'Yes'
            elif expname=='ncrf_60h':
                expstart,rsindx,removetimes,sen_exp=96+86,58,2,'Yes'
            elif expname=='lwcrf':
                expstart,rsidx,removetimes,sen_exp=96+86+62,34,3,'Yes'
            elif expname=='ctl':
                expstart,rsindx,removetimes,sen_exp=None,None,None,'No'
            
            save_forecast(timestep=timestep,LT=LTobj,expname=expname,savepath=savepath,teMP=[teMP1,teMP2,teMP3,teMP4],setting=[expstart,rsindx,removetimes,sen_exp])
        del teMP1,teMP2,teMP3,teMP4
        gc.collect()
        return None
    
class validate_mlr:
    def __init__(self,mlr_prediction=None,modeltime=None):
        self.mlr_pre=mlr_prediction
        self.real_path=['/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/']
        self.modeltime = modeltime
        
    def real_wind(self):
        def put_dict(self,expname=None):
            uvw_outdict = preproc_maria.preprocess('/scratch/itam/maria/processed/','/scratch/itam/maria/',expname,5,'new').preproc_uvw()
            return uvw_outdict
        self.windctrl = put_dict(self,'ctl')
        self.windncrf60 = put_dict(self,'ncrf_60h')
        return self.windctrl, self.windncrf60
    
    def real_windspd(self,u=None,v=None,timestep=None):
        du = [(u[timestep+int(num)]-u[timestep]).reshape(39,360,167) for num in self.modeltime]
        dv = [(v[timestep+int(num)]-v[timestep]).reshape(39,360,167) for num in self.modeltime]
        wspd = [np.sqrt(duu**2+dvv**2) for duu,dvv in zip(du,dv)]
        return du,dv,wspd
    
    def surfacewspd_mlr(self):
        testtrace = np.stack([self.mlr_pre['dv'][i][0,:,:] for i in range(len(self.mlr_pre['dv']))],axis=0)
        testtrace_u = np.stack([self.mlr_pre['du'][i][0,:,:] for i in range(len(self.mlr_pre['du']))],axis=0)
        testint = []
        for inx,obj in (enumerate(self.modeltime)):
            a,b = (testtrace_u[int(inx),:,:]*(obj*60*60)),(testtrace[int(inx),:,:]*(obj*60*60))
            testint.append(np.sqrt(a*a+b*b))
        return testint