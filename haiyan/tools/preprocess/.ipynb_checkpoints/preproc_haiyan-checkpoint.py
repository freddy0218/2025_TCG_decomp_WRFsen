"""
Preprocessing code for Haiyan EnKF ensemble
class 1: preprocess [for PCA analysis]
class 2: vertical_decomp [for spectral analysis]
"""
import xarray as xr
import numpy as np
from tools import read_and_proc
from scipy.ndimage import gaussian_filter
import gc,glob
from tqdm.auto import tqdm
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
pbar = ProgressBar()                
pbar.register() # global registration

def dummy(inlist=None):
    return inlist

def build_a_xarray_dataset(ds=None,varname=None,varfile=None,dims=None,coords=None):
    """
    Create an xarray dataset to store the WRF variables
    --------------------------------------------------
    Input
    --------------------------------------------------
    ds = Empty xarray dataset
    varname = name of the vars in the dataset
    varfile = files that contain the gridded data
    dims, coords = dimensionality of our data
    --------------------------------------------------
    Output
    --------------------------------------------------
    ds = dataset
    """
    def build_xa(arraydata=None,arrayname=None):
        da = xr.DataArray(data=arraydata,dims=dims,coords=coords,name=arrayname)
        return da
    # Complete the code
    for varobj,varnamen in zip(varfile,varname):
        da = build_xa(varobj,varnamen)
        ds = xr.merge( [ds , da ] )
    return ds

class preprocess:
    def __init__(self,uvtpath=None,originpath=None,expname=None,window=None,addctrlmethod='orig',gaussian=True,sigma=None,surfix=None):
        self.uvtpath=uvtpath
        self.originpath=originpath
        self.expname=expname
        self.window=window
        self.method=addctrlmethod
        self.gaussian=gaussian
        self.sigma=sigma
        self.surfix=surfix
        self.coor = None#xr.open_dataset('/scratch/06040/tg853394/tc/output/redux/maria/ctl/post/U.nc')
    #....................................................................................................................................
    # Helper functions
    #....................................................................................................................................
    def nearest_index(self,array=None,value=None):
        """
        Find the index (idx) of a data point in a matrix (array) that is closest to a target value (value)
        """
        idx = (np.abs(array-value)).argmin()
        return idx
    
    def swap_presaved(self,varname=None):
        """
        Manipulate the shape of the pre-stored lists for radial/tangential winds and theta, 
        so that their shapes are consistent with variables that are read directly from WRF (qv etc.)
        """
        assert ((varname=='urad') | (varname=='vtan') | (varname=='theta') | (varname=='rh')),'wrong variable name!'
        if varname=='urad':
            return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'urad'+'/'+'mem'+str(int(self.expname))+'_'+varname)),0,1)
        elif varname=='vtan':
            return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'vtan'+'/'+'mem'+str(int(self.expname))+'_'+varname)),0,1)
        elif varname=='theta':
            return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'theta'+'/'+'mem'+str(int(self.expname))+'_'+varname)),0,1)
        elif varname=='rh':
            return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'rh'+'/'+'mem'+str(int(self.expname))+'_'+varname)),0,1)
    
    def read_azimuth_fields(self,varname=None,wantR=False):
        """
        Read James' azimuthal files.
        Input: varname=variable name
        wantR: boolean for outputing the index corresponding to a radius value
        """
        temp = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(self.originpath+'mem'+str(int(self.expname))+'/azim_'+str(varname)+'_*')[0]],fieldname=[varname])    
        if wantR is True:
            return temp[varname][varname],self.nearest_index(temp[varname][varname].radius,800)
        else:
            return temp[varname][varname]
        
    def stickCTRL(self,ctrlvar=None,senvar=None,method='orig'):
        """
        Force sensitivity experiments to have the same time series lengths to CTRL experiments [Application only to Maria]
        """
        if method=='orig':
            return read_and_proc.add_ctrl_before_senstart(ctrlvar,senvar,self.expname,'Yes')
        else:
            return read_and_proc.add_ctrl_before_senstart_ctrlbase(ctrlvar,senvar,self.expname,'Yes')
        
    def smooth_to_dict(self,var=None,varname=None,window=None):
        ass2 = {}
        for indx,name in enumerate(varname):
            if self.gaussian:
                temp = delayed(self.smooth_array_with_gaussian)(var[indx],0,self.sigma)
            else:
                temp = delayed(self.smooth_array_along_axis)(var[indx],0,window)
            ass2[name] = temp
        assmodel = delayed(dummy)(ass2)
        outdict = assmodel.compute()
        return outdict
    
    def smooth(self,varlist=None):
        return [gaussian_filter(var,sigma=self.sigma) for var in varlist]
    
    def to_dict(self,var=None,varname=None):
        outdict = {}
        for indx,name in enumerate(varname):
            outdict[name] = var[indx]
        return outdict
    
    def smooth_array_along_axis(self,array=None,axis=None,window=3):
        if axis==0:
            aN = array.copy()
        else:
            aN = np.swapaxes(array,axis,0)
        aN = aN.reshape((aN.shape[0], -1))
        return np.asarray([np.convolve(aN[:,i],np.ones(window)/window,'same')[0:aN.shape[0]] for i in range(len(aN[0,:]))]).transpose()
    
    def smooth_array_with_gaussian(self,array=None,axis=None,sigma=[3,0,0,0]):
        if axis==0:
            aN = array.copy()
        else:
            aN = np.swapaxes(array,axis,0)
        temp = gaussian_filter(aN,sigma=sigma)
        return temp.reshape((temp.shape[0], -1))
    
    def forward_diff(self,arrayin=None,delta=None,axis=None,LT=1):
        result = []
        if axis==0:
            for i in range(0,arrayin.shape[axis]-LT):
                temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
                result.append(temp)
            return np.asarray(result)

    #....................................................................................................................................
    # Preprocess
    #....................................................................................................................................
    def preproc_uvwF(self,smooth=True,rmax=None):
        # Read vars
        urad,vtan=(self.swap_presaved('urad')[:,:,:,0:int(rmax)]),(self.swap_presaved('vtan')[:,:,:,0:int(rmax)])
        urad,vtan=da.from_array(np.nan_to_num(urad,nan=np.nanmean(urad))),da.from_array(np.nan_to_num(vtan,nan=np.nanmean(vtan)))
        theta=(self.swap_presaved('theta')[:,:,:,0:int(rmax)])
        theta=da.from_array(np.nan_to_num(theta,nan=np.nanmean(theta)))
        rh=(self.swap_presaved('rh')[:,:,:,0:int(rmax)])
        rh=da.from_array(np.nan_to_num(rh,nan=np.nanmean(rh)))
        
        w,r500=self.read_azimuth_fields(varname='W',wantR=True)
        
        qv = (self.read_azimuth_fields('QVAPOR',False).data)[:,:,:,0:int(rmax)]
        qv = np.nan_to_num(qv,nan=np.nanmean(qv))
        qv,hdia,rthratlw,rthratsw=da.from_array(qv),\
        self.read_azimuth_fields('H_DIABATIC',False),self.read_azimuth_fields('RTHRATLW',False),self.read_azimuth_fields('RTHRATSW',False)
        rthratlwc,rthratswc = self.read_azimuth_fields('RTHRATLWC',False),self.read_azimuth_fields('RTHRATSWC',False)
        # Heat forcing sum
        heatsum = hdia+rthratlw+rthratsw
        rad = rthratlw+rthratsw
        ir = rthratlw+rthratsw-rthratlwc-rthratswc
        del rthratlw,rthratsw,rthratlwc,rthratswc
        gc.collect()
        
        wr = np.nan_to_num(w[:,:,:,0:int(rmax)],nan=np.nanmean(w[:,:,:,0:int(rmax)]))
        heatsumr = np.nan_to_num(heatsum[:,:,:,0:int(rmax)],nan=np.nanmean(heatsum[:,:,:,0:int(rmax)]))
        hdiar = np.nan_to_num(hdia[:,:,:,0:int(rmax)],nan=np.nanmean(hdia[:,:,:,0:int(rmax)]))
        radr = np.nan_to_num(rad[:,:,:,0:int(rmax)],nan=np.nanmean(rad[:,:,:,0:int(rmax)]))
        irr = np.nan_to_num(ir[:,:,:,0:int(rmax)],nan=np.nanmean(ir[:,:,:,0:int(rmax)]))
        wr,heatsumr,hdiar,radr,irr = da.from_array(wr),da.from_array(heatsumr),da.from_array(hdiar),da.from_array(radr),da.from_array(irr)
        del w,heatsum,rad,hdia,ir
        gc.collect()
        
        outdict=self.smooth_to_dict([urad,vtan,wr,qv,theta,heatsumr,hdiar,radr,irr,rh],['u','v','w','qv','theta','heatsum','hdia','rad','ir','rh'],self.window)
        del urad,vtan,qv,wr,heatsumr,hdiar,radr,irr
        gc.collect()
            
        folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/'
        if smooth is True:
            read_and_proc.save_to_pickle(folderpath+'uvwheat/mem'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
        else:
            read_and_proc.save_to_pickle(folderpath+'uvwheat/mem'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
        #print("---Finish!---")
        return None
    
    def preproc_dudvdw(self,smooth=True,rmax=None):
        urad,vtan=(self.swap_presaved('urad')[:,:,:,0:int(rmax)]),(self.swap_presaved('vtan')[:,:,:,0:int(rmax)])
        urad,vtan=da.from_array(np.nan_to_num(urad,nan=np.nanmean(urad))),da.from_array(np.nan_to_num(vtan,nan=np.nanmean(vtan)))
        theta=(self.swap_presaved('theta')[:,:,:,0:int(rmax)])
        theta=da.from_array(np.nan_to_num(theta,nan=np.nanmean(theta)))
        
        w,r500=self.read_azimuth_fields(varname='W',wantR=True)
        wr = np.nan_to_num(w[:,:,:,0:int(rmax)],nan=np.nanmean(w[:,:,:,0:int(rmax)]))
        
        outdict=self.smooth_to_dict([urad,vtan,wr,theta],['u','v','w','theta'],self.window)
        durad,dvtan,dw,dtheta=self.forward_diff(arrayin=outdict['u'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['v'],delta=60*60,axis=0),\
        self.forward_diff(arrayin=outdict['w'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0)
        
        durad = np.nan_to_num(durad)
        dvtan = np.nan_to_num(dvtan)
        dw = np.nan_to_num(dw)
        dtheta = np.nan_to_num(dtheta)
        outdictuvw = {'du':durad,'dv':dvtan,'dw':dw,'dtheta':dtheta}
        
        folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/'
        if smooth is True:
            read_and_proc.save_to_pickle(folderpath+'uvwheat/dudvdwdth/mem'+str(self.expname)+'_smooth_'+self.surfix,outdictuvw,'PICKLE')
        else:
            read_and_proc.save_to_pickle(folderpath+'uvwheat/dudvdwdth/mem'+str(self.expname)+'_'+self.surfix,outdictuvw,'PICKLE')  
        print("---Finish!---")
        return None
    
    def preproc_radcomps(self,smooth=True,rmax=None):
        # Read vars
        _,r500=self.read_azimuth_fields(varname='W',wantR=True)
        hdia,rthratlw,rthratsw=self.read_azimuth_fields('H_DIABATIC',False),self.read_azimuth_fields('RTHRATLW',False),self.read_azimuth_fields('RTHRATSW',False)
        rthratlwc,rthratswc = self.read_azimuth_fields('RTHRATLWC',False),self.read_azimuth_fields('RTHRATSWC',False)
        # Heat forcing sum
        #rad = rthratlw+rthratsw
        irlw,irsw = rthratlw-rthratlwc,rthratsw-rthratswc
        
        rthratlwr = np.nan_to_num(rthratlw[:,:,:,0:int(rmax)],nan=np.nanmean(rthratlw[:,:,:,0:int(rmax)]))
        rthratswr = np.nan_to_num(rthratsw[:,:,:,0:int(rmax)],nan=np.nanmean(rthratsw[:,:,:,0:int(rmax)]))
        rthratlwcr = np.nan_to_num(rthratlwc[:,:,:,0:int(rmax)],nan=np.nanmean(rthratlwc[:,:,:,0:int(rmax)]))
        rthratswcr = np.nan_to_num(rthratswc[:,:,:,0:int(rmax)],nan=np.nanmean(rthratswc[:,:,:,0:int(rmax)]))
        irlwr = np.nan_to_num(irlw[:,:,:,0:int(rmax)],nan=np.nanmean(irlw[:,:,:,0:int(rmax)]))
        irswr = np.nan_to_num(irsw[:,:,:,0:int(rmax)],nan=np.nanmean(irsw[:,:,:,0:int(rmax)]))
        rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr = da.from_array(rthratlwr),da.from_array(rthratswr),da.from_array(rthratlwcr),da.from_array(rthratswcr),da.from_array(irlwr),da.from_array(irswr)
        del rthratlw,rthratsw,rthratlwc,rthratswc,irlw,irsw
        gc.collect()
        
        outdict=self.smooth_to_dict([rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr],['LW','SW','LWC','SWC','IRLW','IRSW'],self.window)
        del rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr
        gc.collect()
            
        folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/'
        if smooth is True:
            read_and_proc.save_to_pickle(folderpath+'uvwheat/mem'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
        else:
            read_and_proc.save_to_pickle(folderpath+'uvwheat/mem'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
        #print("---Finish!---")
        return None

class vertical_decomp:
    def __init__(self,arraysize=39,sincomp=[0.5,-1,-1.5,-2],coorpres=None,kernIN=None):
        #self.input_data = (input_data).copy()
        self.coorpres = np.flipud(coorpres.data)
        self.arraysize = arraysize
        self.sincomp = sincomp
        self.dx=coorpres.data[1]-coorpres.data[0]
        self.kernIN = kernIN
        
    def sin_series(self):
        kern = [np.sin(i*np.linspace(0,2*np.pi,self.arraysize)) for i in self.sincomp]
        return kern #vstack().transpose()
    
    def out_kernpres(self):
        kern = [np.sin(i*np.linspace(0,2*np.pi,self.arraysize)) for i in self.sincomp]
        return kern,self.coorpres 
    
    def get_coeff(self,inputdata=None,allout=True,kern=None):
        # Fourier sine series
        if kern:
            kern=kern[:]
        else:
            kern = [(obj) for obj in self.sin_series()]
        # Output
        coor = [np.nanmean(inputdata)]
        inputt = ((inputdata)-np.nanmean(inputdata)).copy()
        tempcoor = np.trapz(inputt*np.flipud(kern[0]),dx=self.dx)*2/(np.max(self.coorpres)-np.min(self.coorpres))
        coor.append(tempcoor)
        tempcoor2 = np.trapz((inputt-tempcoor*kern[0])*np.flipud(kern[1]),dx=self.dx)*2/(np.max(self.coorpres)-np.min(self.coorpres))
        coor.append(tempcoor2)
        tempcoor3 = np.trapz((inputt-tempcoor*kern[0]-tempcoor2*kern[1])*np.flipud(kern[2]),dx=self.dx)*2/(np.max(self.coorpres)-np.min(self.coorpres))
        coor.append(tempcoor3)
        tempcoor4 = np.trapz((inputt-tempcoor*kern[0]-tempcoor2*kern[1]-tempcoor3*kern[2])*np.flipud(kern[3]),dx=self.dx)*2/(np.max(self.coorpres)-np.min(self.coorpres))
        coor.append(tempcoor4)
        
        if allout:
            return coor,kern,self.coorpres
        else:
            return coor
        
    def do_proc(self,array=None,kern=None):
        return [self.get_coeff((array[:,indx]),False,kern) for indx in (range(len(array[0,:])))]
    
    def proc_on_array(self,array=None,printnum=10000000,kern=None):
        ass2 = delayed(self.do_proc)(array,kern)
        assmodel = delayed(dummy)(ass2)
        return assmodel.compute()     
    
    def process(self,array=None,kern=None):
        temparray = np.asarray([array[:,i,:,:].flatten(order='C') for i in range(self.coorpres.shape[0])])
        #tempcalc = delayed(self.proc_on_array)(da.from_array(temparray[:,:]),1e8,da.from_array(kern))
        #calcmodel = tempcalc.compute()
        calcmodel = np.asarray(self.proc_on_array(array=da.from_array(temparray[:,:]),printnum=10000000,kern=kern))
        return calcmodel
    
    def processFFT(self,array=None,vertlv_num=None):
        assert array.shape[1]==vertlv_num, 'axis shape error'
        calcmodel = np.fft.fft(array,n=20,axis=1)
        return calcmodel[:,0:5,:,:]    