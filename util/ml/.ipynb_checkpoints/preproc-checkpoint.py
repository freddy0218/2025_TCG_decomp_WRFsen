import os,sys
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
import gc
import torch

def prepare_tensors(X,y,notensor='No'):
    X_totrain,X_tovalid,X_totest = X['train'],X['valid'],X['test']
    y_totrain,y_tovalid,y_totest = y['train']*(60*60),y['valid']*(60*60),y['test']*(60*60)
    if notensor=='No':
        calc_device = 'cpu'
        train_Xtensor = torch.FloatTensor(X_totrain).to(calc_device)
        train_ytensor = torch.FloatTensor(y_totrain).to(calc_device)
        val_Xtensor = torch.FloatTensor(X_tovalid).to(calc_device)
        val_ytensor = torch.FloatTensor(y_tovalid).to(calc_device)
        test_Xtensor = torch.FloatTensor(X_totest).to(calc_device)
        test_ytensor = torch.FloatTensor(y_totest).to(calc_device)
        train_data = torch.utils.data.TensorDataset(train_Xtensor, train_ytensor)
        val_data = torch.utils.data.TensorDataset(val_Xtensor, val_ytensor)
        test_data = torch.utils.data.TensorDataset(test_Xtensor, test_ytensor)   
        return train_data,val_data,test_data
    elif notensor=='Yes':
        return {'train':[X_totrain, y_totrain],'valid':[X_tovalid, y_tovalid],'test':[X_totest, y_totest]}
        
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
    
class producePCA:
    def __init__(self,PCATYPE='varimax',n_comps=60):
        self.PCATYPE=PCATYPE
        self.n_comps=n_comps
    
    def fit_cheap_pca(self,n_batches=None,n_comps=None,var=None):
        from sklearn.decomposition import IncrementalPCA
        inc_pca = IncrementalPCA(n_components=n_comps)
        for X_batch in (np.array_split(var.data,n_batches)):
            inc_pca.partial_fit(X_batch)
        return inc_pca

    def fit_pca(self,var=None):
        inc_pca = PCA()
        return inc_pca
        
    def fitPCA(self,arrays=None,arrayname=None,n_batches=10):
        """
        arrays: flat arrays to perform PCs
        arrayname: name of the variables
        axi: 2D or 3D
        """
        PCAdict = {}
        for ind,vnme in tqdm(enumerate(arrayname)):
            if self.PCATYPE=='varimax':
                try:
                    todo = arrays[ind]#-np.mean(arrays[ind])
                    PCAdict[vnme] = CustomPCA(n_components=self.n_comps,rotation='varimax').fit(todo)
                except:
                    sys.exit("Did not install R!")
            elif self.PCATYPE=='orig_cheap':
                PCAdict[vnme] = self.fit_cheap_pca(n_batches=n_batches,n_comps=self.n_comps,var=arrays[ind])
            elif self.PCATYPE=='orig':
                PCAdict[vnme] = self.fit_pca(var=arrays[ind])
        return PCAdict

class proc_X:
    def __init__(self,X,PCA):
        self.X=X
        self.PCA=PCA
        
    def myPCA_projection_sen(self,varname,toproj_flatvar,orig_flatvar):
        projvar_transformed = np.dot(toproj_flatvar-np.nanmean(orig_flatvar,axis=0),self.PCA[varname].components_.T)
        return projvar_transformed

    def create_timeseries(self,varname):
        Xtrain,Xvalid,Xtest = self.X['train'], self.X['valid'], self.X['test']
        train = self.PCA[varname].transform(Xtrain[varname])
        valid = self.myPCA_projection_sen(varname,Xvalid[varname],Xtrain[varname])
        test = self.myPCA_projection_sen(varname,Xtest[varname],Xtrain[varname])
        return {'train':train,'valid':valid,'test':test}

    def normalize_timeseries(self,timeseries=None,category='train'):
        #assert timeseries['u'].shape[-1]==26,"var shape error"
        output = np.zeros_like(timeseries[category])
        for le in range(timeseries[category].shape[1]):
            trainmean,trainstd = np.nanmean(timeseries['train'][:,le]), np.nanstd(timeseries['train'][:,le])
            output[:,le] = (timeseries[category][:,le]-trainmean)/trainstd
        return output
