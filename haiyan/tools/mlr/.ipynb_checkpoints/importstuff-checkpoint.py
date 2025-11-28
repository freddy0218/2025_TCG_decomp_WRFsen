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
from tools import derive_var,read_and_proc,preproc_noensemble,plotting
from tools.mlr import mlr,proc_mlrfcst,maria_IO,cartesian_retrieve
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan

def import_Pipeline_validate(case='Maria',TYPE='3D' or '2D',suffix=None):    
    if case=='Maria':
        filepath = input('Enter Path:')
        #------------------------------------------------------------------------------------------------------------------------------------------
        # Import Preprocessed Flat Data and PCA dictionary
        #------------------------------------------------------------------------------------------------------------------------------------------
        if TYPE=='3D':
            tempdict = {}
            for varname in ['u','v','w','theta']:
                temp = [read_and_proc.depickle(filepath+str(lime)+suffix)[varname] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
                tempdict[varname] = maria_IO.long_MariaExps(temp)[0]
        elif TYPE=='2D':
            tempdict = {}
            size = []
            for varname in ['u','v','w','theta']:
                temp = [read_and_proc.depickle(filepath+str(lime)+suffix)[varname] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
                tempdict[varname] = maria_IO.long_MariaExps(maria_IO.to_azim(temp))[0]
                if varname=='theta':
                    size.append(maria_IO.long_MariaExps(maria_IO.to_azim(temp))[1])
        #------------------------------------------------------------------------------------------------------------------------------------------
        # Save to xarray
        #------------------------------------------------------------------------------------------------------------------------------------------
        dims = ['sample','flatarray']
        coords = dict(sample=np.linspace(0,tempdict['u'].shape[0]-1,tempdict['u'].shape[0]),flatarray=np.linspace(0,tempdict['u'].shape[1]-1,tempdict['u'].shape[1]))
        ds = xr.Dataset(coords=coords)
        maria_data=preproc_haiyan.build_a_xarray_dataset(ds=ds,varname=['u','v','w','theta'],\
                                                         varfile=[tempdict['u'],tempdict['v'],tempdict['w'],tempdict['theta']],dims=dims,coords=coords)
        del tempdict
        gc.collect()
        #------------------------------------------------------------------------------------------------------------------------------------------
        # Import PCA dictionary
        #------------------------------------------------------------------------------------------------------------------------------------------
        if TYPE=='3D':
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/maria/processed/'
            dict1 = read_and_proc.depickle(folderpath+'PCA/PCAdict2')
        elif TYPE=='2D':
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/maria/processed/intermediate/'
            dict1 = read_and_proc.depickle(folderpath+'PCA/PCAdict')
        return maria_data, dict1, size
    elif case=='Haiyan':
        if TYPE=='2D':
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/intermediate/'
            dict1 = read_and_proc.depickle(folderpath+'pca/PCA'+'_'+'dict2_g')
        return dict1