import glob,os,sys
import numpy as np
from tools.validation import r2_analysis
import pandas as pd
from tqdm.auto import tqdm
import gc
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
sys.path.insert(2, '../')
import read_stuff as read
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def flatten(l):
    return [item for sublist in l for item in sublist]

def _get_exp_name(folderpath=None,splitnum=None,folder=2,TYPE='varimax'):
    if TYPE=='varimax':
        return sorted(glob.glob(folderpath+'varimaxpca/X/random/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'varimaxpca/X/random/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='orig':
        return sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='keras':
        return sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='fixTEST':
        return sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')

class proc_inputoutput:
    def __init__(self,validindices=None,testindices=None,pcastorepath=None):
        self.validindices=validindices
        self.testindices=testindices
        self.pcastorepath=pcastorepath #'./store/pca/'
        
    def forward_diff(self,arrayin=None,delta=None,axis=None,LT=1):
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
        
    def _get_time_diff_ts(self,array=None,timedelta=60*60,LT=None):
        store = []
        for exp in array: 
            a = self.forward_diff(exp,timedelta,0,LT)
            if a.shape[0]>0:
                azero = np.zeros((LT))
                store.append(np.concatenate((a,azero),axis=0))
            else:
                store.append(np.zeros((exp.shape[0])))
        return store
    
    def myPCA_projection_sen(self,pca_dict=None,varname=None,toproj_flatvar=None,orig_flatvar=None):
        projvar_transformed = np.dot(toproj_flatvar-np.nanmean(orig_flatvar,axis=0),pca_dict[varname].components_.T)
        return projvar_transformed

    def create_timeseries(self,var=None,varname=None,splitnum=None):
        Xtrain,Xvalid,Xtest = preproc.train_valid_test(var,self.validindices,self.testindices)#[int(self.indices[splitnum][0]),int(self.indices[splitnum][1])],[int(self.indices[splitnum][2]),int(self.indices[splitnum][3])],'Yes')
        pca = read_and_proc.depickle(glob.glob(self.pcastorepath+str(varname)+'/'+str(splitnum)+'/*')[0])
        train = pca[varname].transform(Xtrain)
        valid = self.myPCA_projection_sen(pca,varname,Xvalid,Xtrain)
        test = self.myPCA_projection_sen(pca,varname,Xtest,Xtrain)
        timeseries = {'train':train,'valid':valid,'test':test}
        return timeseries
    
    def normalize_timeseries(self,timeseries=None,category='train'):
        #assert timeseries['u'].shape[-1]==26,"var shape error"
        output = np.zeros_like(timeseries[category])
        for le in range(timeseries[category].shape[1]):
            trainmean,trainstd = np.nanmean(timeseries['train'][:,le]), np.nanstd(timeseries['train'][:,le])
            output[:,le] = (timeseries[category][:,le]-trainmean)/trainstd
        return output
    
    def normalize_timeseries_sensitivity(self,timeseries=None,category='train'):
        #assert timeseries['u'].shape[-1]==26,"var shape error"
        output = np.zeros_like(timeseries[category])
        for le in range(timeseries[category].shape[1]):
            trainmean,trainstd = np.nanmean(timeseries['train'][:,le]), np.nanstd(timeseries['train'][:,le])
            output[:,le] = (timeseries[category][:,le]-trainmean)/trainstd
        return output
    
    def train_valid_test(self,listt=None,splitnum=None):
        #valid, test = [listt[i] for i in [int(self.indices[splitnum][0]),int(self.indices[splitnum][1])]], [listt[i] for i in [int(self.indices[splitnum][2]),int(self.indices[splitnum][3])]]
        valid, test = [listt[i] for i in self.validindices], [listt[i] for i in self.testindices]
        #popindex = [int(self.indices[splitnum][0]),int(self.indices[splitnum][1])]+[int(self.indices[splitnum][2]),int(self.indices[splitnum][3])]
        popindex = self.validindices+self.testindices
        train = [listt[i] for i in range(len(listt)) if i not in popindex]
        return train, valid, test
    
    def _back_to_exp(self,timeseries=None,divider=None):
        if len(timeseries.shape)==2:
            printout = [timeseries[0:divider[0],:]]
            for i in range(1,len(divider)-2):
                printout.append(timeseries[divider[i-1]:divider[i],:])
            printout.append(timeseries[divider[-2]:,:])
        elif len(timeseries.shape)==1:
            printout = [timeseries[0:divider[0]]]
            for i in range(1,len(divider)-2):
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
    
    def create_X(self,vardicts=None,nummem=None,varnames=None,splitnum=None):
        trains,valids,tests = {},{},{}
        for ind,obj in enumerate(varnames):
            timeseries = self.create_timeseries(vardicts[obj],obj,splitnum)
            trains[obj] = self.normalize_timeseries(timeseries,'train')[:,:nummem[ind]]
            valids[obj] = self.normalize_timeseries(timeseries,'valid')[:,:nummem[ind]]
            tests[obj] = self.normalize_timeseries(timeseries,'test')[:,:nummem[ind]]
        return trains,valids,tests 
        
def get_max_intensity(u=None,v=None,shape=[10,360,208]):
    TEMPts = np.max(np.mean(np.sqrt(v.reshape(v.shape[0],shape[0],shape[1],shape[2])[:,0,...]**2+u.reshape(u.shape[0],shape[0],shape[1],shape[2])[:,0,...]**2),axis=1),axis=1)
    return TEMPts

def train_valid_test(expvarlist=None,validindex=None,testindex=None,concat='Yes'):
    X_valid, X_test = [expvarlist[i] for i in validindex], [expvarlist[i] for i in testindex]
    X_traint = expvarlist.copy()
    popindex = validindex+testindex
    X_train = [X_traint[i] for i in range(len(X_traint)) if i not in popindex]
    #assert len(X_train)==16, 'wrong train-valid-test separation!'
    if concat=='Yes':
        return np.concatenate([X_train[i] for i in range(len(X_train))],axis=0), np.concatenate([X_valid[i] for i in range(len(X_valid))],axis=0), np.concatenate([X_test[i] for i in range(len(X_test))],axis=0)
    else:
        return X_train, X_valid, X_test
        
def real_random(folderpath=None,index=None,folder=2,TYPE=None,yfolder=None):
    toextract = _get_exp_name(folderpath,index,folder,TYPE)[0]
    # X
    if TYPE=='varimax':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'varimaxpca/X/random/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'varimaxpca/y/random/*'+str(toextract)+'*'))
    elif TYPE=='orig':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    elif TYPE=='keras':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'keras/y/random/'+str(yfolder)+'/*'+str(toextract)+'*'))
    elif TYPE=='fixTEST':
        Xtrainpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain'+str(toextract)+'*'))
        Xvalidpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xvalid'+str(toextract)+'*'))
        Xtestpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtest'+str(toextract)+'*'))
        #Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'keras/ynew/'+str(yfolder)+'/allY'+str(toextract)+'*'))
    
    Xtest,Xtrain,Xvalid = [read_and_proc.depickle(obj) for obj in [Xtestpath[0],Xtrainpath[0],Xvalidpath[0]]]
    yall = read_and_proc.depickle(yallpath[0])
    return Xtest,Xtrain,Xvalid,yall

def real_random_y(folderpath=None,index=None,folder=2,TYPE=None,yfolder=None):
    toextract = _get_exp_name(folderpath,index,folder,TYPE)[0]
    # X
    if TYPE=='orig':
        yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    elif TYPE=='keras':
        yallpath = sorted(glob.glob(folderpath+'keras/y/random/'+str(yfolder)+'/*'+str(toextract)+'*'))
    elif TYPE=='fixTEST':
        yallpath = sorted(glob.glob(folderpath+'keras/ynew/'+str(yfolder)+'/tsY'+str(toextract)+'*'))
    yall = read_and_proc.depickle(yallpath[0])
    return yall

def delete_padding(inTS=None,outTS=None):
    output_nozero,input_nozero = [],[]
    if len(outTS.shape)>1:
        for i in range(len(outTS[:,0])):
            temp = outTS[i,:]
            tempin = inTS[i,:]
            if temp.all()==0:
                continue
            else:
                output_nozero.append(temp)
                input_nozero.append(tempin)
        return input_nozero,output_nozero
    else:
        for i in range(len(outTS[:])):
            temp = outTS[i]
            tempin = inTS[i,:]
            if temp.all()==0:
                continue
            else:
                output_nozero.append(temp)
                input_nozero.append(tempin)
        return input_nozero,output_nozero 
    
from tools.validation import r2_analysis
class train_optimizedMLR:
    def __init__(self,folderpath=None,modelpath=None,subfoldername=None,ysubfoldername='rh',twoDthreeD='2D' or '3D'):
        self.pcapath=folderpath
        self.modelpath=modelpath
        if twoDthreeD=='2D':
            self.pcastore = read_and_proc.depickle(self.pcapath+'PCAdict2D.pkg')
            self.flatarray = read_and_proc.depickle(self.pcapath+'flatarrays2D.pkg')
        elif twoDthreeD=='3D':
            self.pcastore = read_and_proc.depickle(self.pcapath+'PCAdict3D.pkg')
            self.flatarray = read_and_proc.depickle(self.pcapath+'flatarrays3D.pkg')            
        self.subfoldername=subfoldername
        self.ysubfoldername=ysubfoldername
        self.twoDthreeD=twoDthreeD
        
    def read_Xy(self,subfolders='keras',num=33,needorig='No',onlyY='No' or 'Yes'):
        """
        Read in the processed PC loading time series
        """
        if onlyY=='No':
            Xtest,Xtrain,Xvalid = [],[],[]
            yall = []
            for i in tqdm(range(num)):
                temp1,temp2,temp3,temp4 = real_random(self.modelpath,i,self.subfoldername,subfolders,self.ysubfoldername)
                Xtest.append(temp1)
                Xtrain.append(temp2)
                Xvalid.append(temp3)
                yall.append(temp4)
                
            if needorig=='Yes':
                self.subfoldername=3
                yall_orig = []
                for i in tqdm(range(num)):
                    temp1,temp2,temp3,temp4 = real_random(self.modelpath,i,self.subfoldername,'orig')
                    yall_orig.append(temp4)
                return Xtrain,Xvalid,Xtest,yall,yall_orig
            else:
                return Xtrain,Xvalid,Xtest,yall
        elif onlyY=='Yes':
            yall = []
            for i in tqdm(range(num)):
                yall.append(real_random_y(self.modelpath,i,self.subfoldername,subfolders,self.ysubfoldername))
            return yall
                
    def delete_padding(self,inTS=None,outTS=None):
        output_nozero,input_nozero = [],[]
        if len(outTS.shape)>1:
            for i in range(len(outTS[:,0])):
                temp = outTS[i,:]
                tempin = inTS[i,:]
                if temp.all()==0:
                    continue
                else:
                    output_nozero.append(temp)
                    input_nozero.append(tempin)
            return input_nozero,output_nozero
        else:
            for i in range(len(outTS[:])):
                temp = outTS[i]
                tempin = inTS[i,:]
                if temp.all()==0:
                    continue
                else:
                    output_nozero.append(temp)
                    input_nozero.append(tempin)
            return input_nozero,output_nozero    
        
    def y_truth(self,divider=None,lti=24,num=33,withW=True,splitnum=None):
        if withW is True:
            temp = [r2_analysis.preproc_r2(self.flatarray,None,None)._back_to_exp(timeseries=self.flatarray[varname],divider=divider) for varname in ['u','v','w','theta']]
        else:
            temp = [r2_analysis.preproc_r2(self.flatarray,None,None)._back_to_exp(timeseries=self.flatarray[varname],divider=divider) for varname in ['u','v','theta']]
        train_realUV,valid_realUV,test_realUV = [],[],[]
        for ind,obj in tqdm(enumerate(splitnum)):#range(num)):#range(15)):#range(1)):
            try:
                tempindex = _get_exp_name(self.modelpath,obj,3,'orig')[1]
            except:
                tempindex = _get_exp_name('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/intermediate/',obj,3,'orig')[1]                
            validindex,testindex = [int(tempindex[0]),int(tempindex[1])],[int(tempindex[2]),int(tempindex[3])]
            trainobj = r2_analysis.preproc_r2(self.flatarray,validindex,testindex).windrates_real(uvwheat=temp,LT=lti,category='train',withW=withW,twoDthreeD=self.twoDthreeD)
            validobj = r2_analysis.preproc_r2(self.flatarray,validindex,testindex).windrates_real(uvwheat=temp,LT=lti,category='valid',withW=withW,twoDthreeD=self.twoDthreeD)
            testobj = r2_analysis.preproc_r2(self.flatarray,validindex,testindex).windrates_real(uvwheat=temp,LT=lti,category='test',withW=withW,twoDthreeD=self.twoDthreeD)
            train_realUV.append(trainobj)
            valid_realUV.append(validobj)
            test_realUV.append(testobj)
        del trainobj,validobj,testobj
        gc.collect()
        return {'train':train_realUV,'valid':valid_realUV,'test':test_realUV}
    
    def _where_exp_MLRpred(self,splitnum=None,subfolders='fixTEST',divider=None,exp_index=None,LT=24):
        orig = np.r_[divider[0], np.diff(divider)]
        getindex = [int(obj) for obj in _get_exp_name(self.modelpath,splitnum,self.subfoldername,subfolders)[1]]
        if exp_index not in getindex:
            numexpout = sum([int(obj)<exp_index for obj in (getindex)])
            myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i not in getindex]).cumsum()
            return myindices[exp_index-numexpout-1],myindices[exp_index-numexpout],myindices[exp_index-numexpout]-myindices[exp_index-numexpout-1],'train'
        else:
            myidex = getindex.index(exp_index)
            if myidex<=1:
                category='valid'
                myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i in getindex[0:2]]).cumsum()
                if myidex==0:
                    return 0,myindices[0],myindices[0],category
                elif myidex==1:
                    return myindices[0],myindices[1],myindices[1]-myindices[0],category
            else:
                category='test'
                myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i in getindex[2:4]]).cumsum()
                if myidex==2:
                    return 0,myindices[0],myindices[0],category
                elif myidex==3:
                    return myindices[0],myindices[1],myindices[1]-myindices[0],category
                
    def where_exp_MLRpred(self,divider=None,num=40,expnum=10,LT=24):
        start,end,exp,size = [],[],[],[]
        for i in range(int(num)):
            temp1,temp2,temp3,temp4 = self._where_exp_MLRpred(splitnum=i,divider=divider,exp_index=expnum,LT=LT)
            start.append(temp1)
            end.append(temp2)
            exp.append(temp4)
            size.append(temp3)
            #except:
            #    start.append(None)
            #    end.append(None)
            #    exp.append(None)
            #    size.append(None)
        return pd.DataFrame.from_dict({'start':start,'end':end,'exp':exp,'size':size})

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
        
def tensor_prepare(WSPDpath=None,Xpath=None,validindex=None,testindex=[10,17],shuffle=True,leave100=True):
    """
    Prepare data for training ML models. 
    WSPDpath: Path to access maximum surface winds data
    Xpath: Path to access longwave radiative heating data
    validindex: Index for ensemble members in the validation set
    testindex: Index for ensemble members in the test set (10,17)
    shuffle: Shuffle the data (False for inference)
    leave100: leave out data at 100 hPa (reason: found occasion artifacts causing validation to be outside the distribution of train data)
    """
    # Read files
    storeWSPD = read_and_proc.depickle(WSPDpath)
    Xtrain_cart = read_and_proc.depickle(Xpath)['train']
    Xvalid_cart = read_and_proc.depickle(Xpath)['valid']
    Xtest_cart = read_and_proc.depickle(Xpath)['test']
    # Separate wind speed data to create outputs
    WSPD_train,WSPD_valid,WSPD_test = train_valid_test(storeWSPD,validindex,testindex,'No')
    # Values for output normalization
    wspdmean,wspdstd = np.nanmean(np.concatenate(WSPD_train)),np.nanstd(np.concatenate(WSPD_train))
    y = {'train':[_get_time_diff_ts(WSPD_train,60*60,int(LDTobj)) for LDTobj in np.linspace(0,35,36)+1],\
         'valid':[_get_time_diff_ts(WSPD_valid,60*60,int(LDTobj)) for LDTobj in np.linspace(0,35,36)+1],\
         'test':[_get_time_diff_ts(WSPD_test,60*60,int(LDTobj)) for LDTobj in np.linspace(0,35,36)+1]}

    ytrain = np.concatenate(y['train'][23])
    yvalid = np.concatenate(y['valid'][23])
    ytest = np.concatenate(y['test'][23])
    Xtrain_cart_n, ytrain_n = read.delete_padding(Xtrain_cart,ytrain)
    Xvalid_cart_n, yvalid_n = read.delete_padding(Xvalid_cart,yvalid)
    Xtest_cart_n, ytest_n = read.delete_padding(Xtest_cart,ytest)

    del Xtrain_cart,Xvalid_cart,Xtest_cart,ytrain,yvalid,ytest
    gc.collect()

    # ------------------------------- Standardization -------------------------------------------#
    mean = np.asarray(Xtrain_cart_n).mean(axis=0, keepdims=True)  
    std = np.asarray(Xtrain_cart_n).std(axis=0, keepdims=True)    
    X_train_std = (np.asarray(Xtrain_cart_n) - mean) / (std)
    X_val_std = (np.asarray(Xvalid_cart_n) - mean) / (std)
    X_test_std = (np.asarray(Xtest_cart_n) - mean) / (std)

    mean_y = np.asarray(ytrain_n).mean()
    std_y = np.asarray(ytrain_n).std()
    ytrain_std = (np.asarray(ytrain_n) - mean_y)/std_y
    yvalid_std = (np.asarray(yvalid_n) - mean_y)/std_y
    ytest_std = (np.asarray(ytest_n) - mean_y)/std_y

    # Convert your data beforehand
    if leave100:
        train_Xtensor = torch.FloatTensor(np.asarray(X_train_std)[:, np.newaxis, :-1, :, :])
        val_Xtensor = torch.FloatTensor(np.asarray(X_val_std)[:, np.newaxis, :-1, :, :])
        test_Xtensor = torch.FloatTensor(np.asarray(X_test_std)[:, np.newaxis, :-1, :, :])
    else:
        train_Xtensor = torch.FloatTensor(np.asarray(X_train_std)[:, np.newaxis, :, :, :])
        val_Xtensor = torch.FloatTensor(np.asarray(X_val_std)[:, np.newaxis, :, :, :])  
        test_Xtensor = torch.FloatTensor(np.asarray(X_test_std)[:, np.newaxis, :, :, :])
    train_ytensor = torch.FloatTensor(np.asarray(ytrain_std)[:, np.newaxis])
    val_ytensor = torch.FloatTensor(np.asarray(yvalid_std)[:, np.newaxis])
    test_ytensor = torch.FloatTensor(np.asarray(ytest_std)[:, np.newaxis])
    input_shape = train_Xtensor.shape

    train_dataset = TensorDataset(train_Xtensor, train_ytensor)
    val_dataset = TensorDataset(val_Xtensor, val_ytensor)
    test_dataset = TensorDataset(test_Xtensor, test_ytensor)

    del Xtrain_cart_n, ytrain_n, Xvalid_cart_n, yvalid_n, Xtest_cart_n, ytest_n
    gc.collect()

    dataset = TensorDataset(train_Xtensor, train_ytensor)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=shuffle)
    dataset_valid = TensorDataset(val_Xtensor, val_ytensor)
    val_loader = DataLoader(dataset_valid, batch_size=8, shuffle=shuffle)
    dataset_test = TensorDataset(test_Xtensor, test_ytensor)
    test_loader = DataLoader(dataset_test, batch_size=8, shuffle=shuffle)
    return dataset, dataset_valid, dataset_test, train_loader, val_loader, test_loader, train_Xtensor, train_ytensor, val_Xtensor, val_ytensor, test_Xtensor, test_ytensor, input_shape, {'meanX':mean,'meanY':mean_y,'stdX':std,'stdY':std_y}
    
