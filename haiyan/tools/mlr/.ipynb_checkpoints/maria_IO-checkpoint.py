import numpy as np
import gc
from tqdm.auto import tqdm
from tools import read_and_proc

def long_MariaExps(array=None,start=None):
    haiyan_temparray = [array[0][12:,:],array[1][int(start[0]):,:],array[2][int(start[1]):,:],array[3][int(start[2]):,:],array[4][int(start[3]):,:]]
    del array
    gc.collect()
    return np.concatenate(haiyan_temparray,axis=0),[obj.shape for obj in haiyan_temparray]

def to_azim(array=None,shape=[39,360,200]):
    def _to_azim(array=None):
        arrayn = array.reshape(array.shape[0],shape[0],shape[1],shape[2])
        return np.nanmean(arrayn,axis=2).reshape(array.shape[0],shape[0]*shape[2])
    return [_to_azim(obj) for obj in array]

def flatten(t):
    #https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    return [item for sublist in t for item in sublist]

def prepare_inputdataset2(inputTS=None,flat_out=np.zeros((83,12+3+3+9)),lefttimelim=None):
    def flattenlist(inlist=None):
        tempvarlist = []
        for item in inlist:
            tempvarlist.append(item)
        return read_and_proc.flatten(tempvarlist)
    input_dataset36 = []
    for timeseries in inputTS:#[pca_timeseries_36,pcaur_timeseries_36,pcavr_timeseries_36,pcaw_timeseries_36]:
        input_dataset36.append(timeseries)
    for i in (range(inputTS[0][:,0].shape[0])):
        tempinlist = [obj[i,:] for obj in input_dataset36]
        flat_out[i,:] = flattenlist(tempinlist)
    del timeseries,i
    return flat_out

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    if len(arrayin.shape)>1:
        result = []
        if axis==0:
            for i in range(0,arrayin.shape[axis]-LT):
                temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
                result.append(temp)
            return np.asarray(result)
    else:
        result = []
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT]-arrayin[i])/(LT*delta)
            result.append(temp)
        return np.asarray(result)   
    
def myPCA_projection_sen(pca_dict=None,varname=None,toproj_flatvar=None,orig_flatvar=None):
    pca_orig = pca_dict[varname].transform(orig_flatvar)
    if pca_dict[varname].mean_ is not None:
        orig_mean = pca_dict[varname].mean_
    projvar_transformed = np.dot(toproj_flatvar-np.nanmean(toproj_flatvar,axis=0),pca_dict[varname].components_.T)
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
            ts_dict[obj] = self.PCAdict[obj].transform(flatvar[obj].data)[:,0:self.nummem[indx]]
        return ts_dict
    
    def produce_Qsentimeseries(self,senvar_name=None,refvar_name='rad',numQ=None,flatvar=None,senflatvar=None):
        ts_dict = {}
        temp = [myPCA_projection_sen(pca_dict=self.PCAdict,varname=refvar_name,toproj_flatvar=senflatvar[obj].data,orig_flatvar=flatvar[refvar_name].data)[1][:,0:numQ] for indx,obj in enumerate(senvar_name)]
        return dict(zip(senvar_name,temp))
    
    def normalize_timeseries(self,timeseries=None):
        #assert timeseries['u'].shape[-1]==50,"var shape error"
        ts_dict = {}
        for indx,obj in tqdm(enumerate(self.varname)):
            ts_dict[obj] = (timeseries[obj]-np.nanmean(timeseries[obj],axis=0))/np.nanstd(timeseries[obj],axis=0)
        return ts_dict
    
    def normalize_timeseries_decomp(self,sentimeseries=None,reftimeseries=None,senvarnames=None,refvarname='rad'):
        ts_dict = {}
        meanstd_dict = {}
        for indx,obj in tqdm(enumerate(senvarnames)):
            tempf = -np.mean(sentimeseries[obj],axis=0)/np.std(reftimeseries[refvarname],axis=0)
            ts_dict[obj] = (sentimeseries[obj]-np.mean(sentimeseries[obj],axis=0))/np.std(reftimeseries[refvarname],axis=0)
            meanstd_dict[obj] = np.broadcast_to(tempf, (sentimeseries[obj].shape[0], sentimeseries[obj].shape[1]))
        return ts_dict,meanstd_dict
    
    ###################################################################################################################################################
    # Produce Input Dataset
    ###################################################################################################################################################      
    def _back_to_exp(self,timeseries=None,divider=None):
        if len(timeseries.shape)==2:
            printout = [timeseries[0:divider[0],:]]
            for i in range(1,len(divider)-1):
                printout.append(timeseries[divider[i-1]:divider[i],:])
            printout.append(timeseries[divider[-2]:,:])
        elif len(timeseries.shape)==1:
            printout = [timeseries[0:divider[0]]]
            for i in range(1,len(divider)-1):
                printout.append(timeseries[divider[i-1]:divider[i]])
            printout.append(timeseries[divider[-2]:])            
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
    
    def train_valid_test(self,expvarlist=None,validindex=None,testindex=None,concat='Yes'):
        X_valid, X_test = [expvarlist[i] for i in validindex], [expvarlist[i] for i in testindex]
        X_traint = expvarlist.copy()
        popindex = validindex+testindex
        #[X_train.pop(i) for i in validindex]
        #[X_train.pop(i) for i in testindex]
        X_train = [X_traint[i] for i in range(len(X_traint)) if i not in popindex]
        assert len(X_train)==16, 'wrong train-valid-test separation!'
        if concat=='Yes':
            return np.concatenate([X_train[i] for i in range(len(X_train))],axis=0), np.concatenate([X_valid[i] for i in range(len(X_valid))],axis=0), np.concatenate([X_test[i] for i in range(len(X_test))],axis=0)
        else:
            return X_train, X_valid, X_test
    
    def make_X(self,expvarlist=None,varwant=None,validindex=None,testindex=None,concat='Yes'):
        trainlist,validlist,testlist = [],[],[]
        for obj in varwant:
            test1,test2,test3 = self.train_valid_test(expvarlist[obj],validindex,testindex,'Yes')
            trainlist.append(test1)
            validlist.append(test2)
            testlist.append(test3)
        return np.concatenate([trainlist[i] for i in range(len(trainlist))],axis=1), np.concatenate([validlist[i] for i in range(len(validlist))],axis=1), np.concatenate([testlist[i] for i in range(len(testlist))],axis=1)
    
    def make_X_nosep(self,expvarlist=None,varwant=None):
        trainlist = []
        for obj in varwant:
            test1 = np.concatenate([arrays for arrays in expvarlist[obj]], axis=0)
            trainlist.append(test1)
        return np.concatenate([trainlist[i] for i in range(len(trainlist))],axis=1)
    
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
        
        def _get_time_diff_ts(array=None,timedelta=60*60,LT=None):
            store = []
            for exp in array:
                a = forward_diff(exp,timedelta,0,LT)
                if a.shape[0]>0:
                    azero = np.zeros((LT))
                    store.append(np.concatenate((a,azero),axis=0))
                else:
                    store.append(np.zeros((exp.shape[0])))
            return store
        
        if type(inputvar)!=dict:
            return _get_time_diff_ts(array=inputvar,LT=LT)
        else:
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
        test = [self.get_time_diff_terms(inputdict,int(LDTobj)) for LDTobj in LDT]
        return [_make_Y(timediffobj) for timediffobj in test]
    
    def make_Y_nosep(self,inputdict=None,LDT=None):
        def _make_Y(inputt=None):
            varTRAIN = []
            for varobj in ['u','v','w','theta']:
                test1 = np.concatenate([arrays for arrays in inputt[varobj]], axis=0)
                varTRAIN.append(test1)
            return np.concatenate([varTRAIN[i] for i in range(len(varTRAIN))],axis=1)
        test = [self.get_time_diff_terms(inputdict,int(LDTobj)) for LDTobj in LDT]
        return [_make_Y(timediffobj) for timediffobj in test]

    def make_Y_ts(self,inputdict=None,LDT=None):
        test = [self.get_time_diff_terms(inputdict,int(LDT[LDTnum])) for LDTnum in (range(len(LDT)))]
        return test
    
class datacheck:
    def __init__(self,pcadict=None,flatvardict=False,divider=None):
        #self.folderpath=folderpath
        self.pcadict = pcadict
        self.ctlflatvar = flatvardict
        self.divider = divider
        
    def _back_to_exp(self,timeseries=None,divider=None):
        printout = [timeseries[0:divider[0],:]]
        for i in range(1,len(divider)-1):
            printout.append(timeseries[divider[i-1]:divider[i],:])
        printout.append(timeseries[divider[-2]:,:])
        return printout
    
    def dudvdwVAR(self,dudvdw=None,vartest='w'):
        #dudvdw = read_and_proc.depickle(self.folderpath+dudvdwpath)
        timeseries = self.pcadict[vartest].transform(self.ctlflatvar[vartest])
        left_dot = [forward_diff(obj,60*60,0,1) for obj in self._back_to_exp(timeseries,self.divider)]
        left_dott = np.concatenate([obj for obj in left_dot],axis=0)
        for i in np.linspace(0,90,46):
            tempobj = np.dot(left_dott[:,0:int(i)],(self.pcadict[vartest].components_[0:int(i)]))
            print((np.var(tempobj)/np.var(dudvdw['d'+str(vartest)])))
        #TESTu_var = [np.var(obj)/np.var(dudvdw['d'+str(vartest)]) for obj in TESTu]
        del left_dot,left_dott
        gc.collect()
        return None

