from tools import read_and_proc
import gc
from tools.preprocess import myeof
from tqdm import tqdm
from dask import delayed
from sklearn.decomposition import PCA
import numpy as np

def dummy(inlist=None):
    return inlist

class do_PCA:
    def __init__(self,expname=['ctl','ncrf_36h','ncrf_60h','lwcrf'],varname=['u','v','w','qv','heatsum'],timezoom=[24,120],radius=167):
        self.varname=varname
        self.timezoom=timezoom
        self.radius=radius
        self.expname=expname
        
    def fit_PCA(self,array=None):
        skpcaVAR = PCA()
        skpcaVAR.fit(array)
        print('Complete')
        return skpcaVAR
        
    def do_ctl_PCA(self,folderpath=None,suffix=None):
        listdict = [read_and_proc.depickle(folderpath+str(self.expname[i])+'_'+suffix) for i in range(len(self.expname))]
        ctl_var = [listdict[0][strvar][self.timezoom[0]:self.timezoom[1]] for strvar in self.varname]
        
        PCAska = {}
        for inx,obj in (enumerate(ctl_var)):
            pcatemp = delayed(self.fit_PCA)(obj)
            PCAska[self.varname[inx]] = pcatemp
        PCAskaa = delayed(dummy)(PCAska)
        PCAass = PCAskaa.compute()
        read_and_proc.save_to_pickle(folderpath+'/PCA/'+str(self.expname[0])+'_'+'PCA_dict1',PCAass,'PICKLE')
        return PCAass

    def do_PCA_onevar(self,ctlvar=None,folderpath=None):
        ctl_var = [ctlvar[self.timezoom[0]:self.timezoom[1]]]
        PCAska = {}
        for inx,obj in (enumerate(ctl_var)):
            pcatemp = delayed(self.fit_PCA)(obj)
            PCAska[self.varname[inx]] = pcatemp
        PCAskaa = delayed(dummy)(PCAska)
        PCAass = PCAskaa.compute()
        read_and_proc.save_to_pickle(folderpath+'/PCA/'+str(self.varname[0])+'_'+'PCA_dict1',PCAass,'PICKLE')
        return PCAass
    
    def required_components(self,PCAdict=None,varlist=['u','v','w','qv','heatsum'],target=0.9):
        self.components = [np.abs(PCAdict[varlist[i]].explained_variance_ratio_.cumsum()-target).argmin() for i in range(len(varlist))]
        return self.components