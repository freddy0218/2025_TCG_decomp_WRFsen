import numpy as np
import gc
from tools import read_and_proc
from tqdm.auto import tqdm

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
        return np.asarray(result)
    
def myPCA_projection_sen(pca_dict=None,varname=None,toproj_flatvar=None,orig_flatvar=None):
    pca_orig = pca_dict[varname].transform(orig_flatvar)
    if pca_dict[varname].mean_ is not None:
        orig_mean = pca_dict[varname].mean_
    projvar_transformed = np.dot(toproj_flatvar-np.nanmean(orig_flatvar,axis=0),pca_dict[varname].components_.T)
    del orig_mean
    gc.collect()
    return pca_orig, projvar_transformed
    
class input_output:
    def __init__(self,PCAdict=None,folderpath=None,ts_varname=None,nummem=None):
        self.PCAdict = PCAdict
        self.varname=ts_varname
        self.nummem = nummem # u: 36 (40% variability in du), v:16/32 (40% dv var;50%), w:44 (40% dw var)
    
    ###################################################################################################################################################
    # Produce time series
    ###################################################################################################################################################    
    def produce_timeseries(self,flatvar=None):
        ts_dict = {}
        for indx,obj in tqdm(enumerate(self.varname)):
            ts_dict[obj] = self.PCAdict[obj].transform(flatvar[obj].data)[:,0:]
        return ts_dict
    
    def produce_Qsentimeseries(self,senvar_name=None,refvar_name='rad',numQ=None,flatvar=None,senflatvar=None):
        ts_dict = {}
        temp = [myPCA_projection_sen(pca_dict=self.PCAdict,varname=refvar_name,toproj_flatvar=flatvar[obj].data,orig_flatvar=senflatvar[refvar_name].data)[1][:,0:numQ[indx]] for indx,obj in enumerate(senvar_name)]
        return dict(zip(senvar_name,temp))
    
    def normalize_timeseries(self,timeseries=None):
        assert timeseries['u'].shape[-1]==150,"var shape error"
        ts_dict = {}
        for indx,obj in tqdm(enumerate(self.varname)):
            ts_dict[obj] = (timeseries[obj]-np.nanmean(timeseries[obj],axis=0))/np.nanstd(timeseries[obj],axis=0)
        return ts_dict
    
    ###################################################################################################################################################
    # Produce Input Dataset
    ###################################################################################################################################################      
    def _back_to_exp(self,timeseries=None,divider=None):
        printout = [timeseries[0:divider[0],:]]
        for i in range(1,19):
            printout.append(timeseries[divider[i-1]:divider[i],:])
        printout.append(timeseries[divider[-2]:,:])
        return printout
    
    def back_to_exp(self,inputlong=None,divider=None):
        ts_dict = {}
        for indx,obj in tqdm(enumerate(self.varname)):
            ts_dict[obj] = self._back_to_exp(inputlong[obj],divider)
        return ts_dict
    
    def train_valid_test(self,expvarlist=None,validindex=None,testindex=None,concat='Yes'):
        X_valid, X_test = [expvarlist[i] for i in validindex], [expvarlist[i] for i in testindex]
        X_train = expvarlist.copy()
        [X_train.pop(i) for i in validindex]
        [X_train.pop(i) for i in testindex]
        assert len(X_train)==16, 'wrong train-valid-test separation!'
        if concat=='Yes':
            return np.concatenate([X_train[i] for i in range(len(X_train))],axis=0), np.concatenate([X_valid[i] for i in range(len(X_valid))],axis=0), np.concatenate([X_test[i] for i in range(len(X_test))],axis=0)
        else:
            return X_train, X_valid, X_test
    
    def make_X(self,expvarlist=None,varwant=None,validindex=None,testindex=None,concat='Yes'):
        trainlist,validlist,testlist = [],[],[]
        for obj in varwant:
            test1,test2,test3 = self.train_valid_test(exp_pca_norml[obj],validindex,testindex,'Yes')
            trainlist.append(test1)
            validlist.append(test2)
            testlist.append(test3)
        return np.concatenate([trainlist[i] for i in range(len(trainlist))],axis=1), np.concatenate([validlist[i] for i in range(len(validlist))],axis=1), np.concatenate([testlist[i] for i in range(len(testlist))],axis=1)
    
    ###################################################################################################################################################
    # Produce Output Dataset
    ###################################################################################################################################################
    def get_time_diff_terms(self,inputvar=None,LT=None):
        def _get_time_diff(array=None,timedelta=60*60,LT=None):
            store = []
            for exp in array:
                a = forward_diff(exp,timedelta,0,LT)
                if a.shape[0]>0:
                    azero = np.zeros((LT,exp.shape[-1]))
                    store.append(np.concatenate((a,azero),axis=0))
                else:
                    store.append(np.zeros((exp.shape[0],exp.shape[-1])))
            return store
        
        storedict = {}
        for wantvar in ['u','v','w','theta']:
            storedict[wantvar] = _get_time_diff(array=inputvar[wantvar],LT=LT)
        return storedict
    
    def make_Y(self,inputdict=None,LDT=None,validindex=[1,6],testindex=[2,12]):
        def _make_Y(inputt=None):
            varTRAIN,varVALID,varTEST = [],[],[]
            for varobj in ['u','v','w','theta']:
                test1,test2,test3 = self.train_valid_test(expvarlist=inputt[varobj],validindex=validindex,testindex=testindex,concat='Yes')
                varTRAIN.append(test1)
                varVALID.append(test2)
                varTEST.append(test3)
            return np.concatenate([varTRAIN[i] for i in range(len(varTRAIN))],axis=1), np.concatenate([varVALID[i] for i in range(len(varVALID))],axis=1), np.concatenate([varTEST[i] for i in range(len(varTEST))],axis=1)
        
        test = [self.get_time_diff_terms(exp_pca_nonorml,int(LDTobj)) for LDTobj in LDT]
        return [_make_Y(timediffobj) for timediffobj in test]
    
class datacheck:
    def __init__(self,pcadict=None,flatvardict=False):
        #self.folderpath=folderpath
        self.pcadict = pcadict
        self.ctlflatvar = flatvardict
        
    def back_to_exp(self,inputlong=None,divider=None):
        ts_dict = {}
        for indx,obj in tqdm(enumerate(self.varname)):
            ts_dict[obj] = self._back_to_exp(inputlong[obj],divider)
        return ts_dict
    
    def dudvdwVAR(self,dudvdw=None,vartest='w'):
        #dudvdw = read_and_proc.depickle(self.folderpath+dudvdwpath)
        TESTu = [np.dot(forward_diff(self.pcadict[vartest].transform(self.ctlflatvar[vartest])[:,0:int(i)],60*60,0,1),(self.pcadict[vartest].components_[0:int(i)])) for i in np.linspace(0,90,46)]
        TESTu_var = [np.var(obj)/np.var(dudvdw['d'+str(vartest)]) for obj in TESTu]
        del TESTu
        gc.collect()
        return TESTu_var
    
    #def dthdQVAR(self,dudvdwpath=None,vartest='th',smooth24=False):
    #    dudvdw = read_and_proc.depickle(self.folderpath+dudvdwpath)
    #    if vartest=='th':
    #        vartestPC='theta'
    #        if smooth24:
    #            TESTu = [np.dot(forward_diff(self.pcadict[vartestPC].transform(self.ctlflatvar[0][vartestPC])[:,0:int(i)],60*60,0,1),(self.pcadict[vartestPC].components_[0:int(i)])) for i in np.linspace(0,60,31)]
    #        else:
    #            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/'
    #            PCAtheta = read_and_proc.depickle(folderpath+'PCA/theta_PCA_dict1')
    #            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/flatvar/'
    #            thetavar = read_and_proc.depickle(folderpath+'theta_'+'preproc_dict1')
    #            TESTu = [np.dot(forward_diff(PCAtheta[vartestPC].transform(thetavar['ctlTHETA'])[:,0:int(i)],60*60,0,1),(PCAtheta[vartestPC].components_[0:int(i)])) for i in np.linspace(0,60,31)]
    #    elif vartest=='Q':
    #        vartestPC = 'heatsum'
    #        TESTu = [np.dot(forward_diff(self.pcadict[vartestPC].transform(self.ctlflatvar[0][vartestPC])[:,0:int(i)],60*60,0,1),\
    #                        (self.pcadict[vartestPC].components_[0:int(i)])) for i in np.linspace(0,60,31)]
    #    TESTu_var = [np.var(obj)/np.var(dudvdw['d'+str(vartest)]) for obj in TESTu]
    #    del TESTu
    #    gc.collect()
    #    return TESTu_var