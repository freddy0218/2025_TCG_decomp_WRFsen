import xarray as xr
import numpy as np
from tools import read_and_proc
from scipy.ndimage import gaussian_filter
import gc,glob
from tqdm import tqdm
import dask.array as da
from dask import delayed

def dummy(inlist=None):
    return inlist

class preprocess:
    def __init__(self,uvtpath=None,originpath=None,expname=None,window=None,addctrlmethod='orig',gaussian=True,sigma=None,surfix=None,outsize=167):
        self.uvtpath=uvtpath
        self.originpath=originpath
        self.expname=expname
        self.window=window
        self.method=addctrlmethod
        self.gaussian=gaussian
        self.sigma=sigma
        self.surfix=surfix
        self.outsize=outsize
        self.coor = None#xr.open_dataset('/scratch/06040/tg853394/tc/output/redux/maria/ctl/post/U.nc')
    #....................................................................................................................................
    # Helper functions
    #....................................................................................................................................
    def nearest_index(self,array=None,value=None):
        idx = (np.abs(array-value)).argmin()
        return idx
    
    def swap_presaved(self,varname=None,ctl=True):
        """
        Manipulate the shape of the pre-stored lists for radial/tangential winds and theta, 
        so that their shapes are consistent with variables that are read directly from WRF (qv etc.)
        """
        assert ((varname=='urad') | (varname=='vtan') | (varname=='theta')),'wrong variable name!'
        if varname=='urad':
            if ctl is True:
                return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'urad'+'/'+str('ctl')+'_'+varname)),0,1)
            else:
                return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'urad'+'/'+str(self.expname)+'_'+varname)),0,1)
        elif varname=='vtan':
            if ctl is True:
                return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'vtan'+'/'+str('ctl')+'_'+varname)),0,1)
            else:
                return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'vtan'+'/'+str((self.expname))+'_'+varname)),0,1)
        elif varname=='theta':
            if ctl is True:
                return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'theta'+'/'+str('ctl')+'_'+varname)),0,1)
            else:
                return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'theta'+'/'+str((self.expname))+'_'+varname)),0,1)
    
    def read_azimuth_fields(self,varname=None,wantR=False,ctl=True):
        if ctl is True:
            temp = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(self.originpath+'ctl'+'/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
        else:
            temp = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(self.originpath+str(self.expname)+'/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
            
        if wantR is True:
            return temp[varname][varname],self.nearest_index(temp[varname][varname].radius,500)
        else:
            return temp[varname][varname]
        
    def stickCTRL(self,ctrlvar=None,senvar=None,method='orig'):
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
    def preproc_uvwF(self,smooth=True):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        # Read vars
        urad,vtan=da.from_array(self.swap_presaved('urad',TYPE)[:,:,:,0:int(self.outsize)]),da.from_array(self.swap_presaved('vtan',TYPE)[:,:,:,0:int(self.outsize)])
        theta=da.from_array(self.swap_presaved('theta',TYPE)[:,:,:,0:int(self.outsize)])
        w,r500=self.read_azimuth_fields('W',True,TYPE)
        qv,hdia,rthratlw,rthratsw,rthratlwc,rthratswc=da.from_array(self.read_azimuth_fields('QVAPOR',False,TYPE).data)[:,:,:,0:int(self.outsize)],\
        self.read_azimuth_fields('H_DIABATIC',False,TYPE),\
        self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE),\
        self.read_azimuth_fields('RTHRATLWC',False,TYPE),self.read_azimuth_fields('RTHRATSWC',False,TYPE)
        # Heat forcing sum
        heatsum = hdia+rthratlw+rthratsw
        rad = rthratlw+rthratsw
        ir = rthratlw+rthratsw-rthratlwc-rthratswc
        del rthratlw,rthratsw
        gc.collect()
        
        wr = np.nan_to_num(w[:,:,:,0:int(self.outsize)],nan=np.nanmean(w[:,:,:,0:int(self.outsize)]))
        heatsumr = np.nan_to_num(heatsum[:,:,:,0:int(self.outsize)],nan=np.nanmean(heatsum[:,:,:,0:int(self.outsize)]))
        hdiar = np.nan_to_num(hdia[:,:,:,0:int(self.outsize)],nan=np.nanmean(hdia[:,:,:,0:int(self.outsize)]))
        radr = np.nan_to_num(rad[:,:,:,0:int(self.outsize)],nan=np.nanmean(rad[:,:,:,0:int(self.outsize)]))
        irr = np.nan_to_num(ir[:,:,:,0:int(self.outsize)],nan=np.nanmean(ir[:,:,:,0:int(self.outsize)]))
        wr,heatsumr,hdiar,radr,irr = da.from_array(wr),da.from_array(heatsumr),da.from_array(hdiar),da.from_array(radr),da.from_array(irr)
        del w,heatsum,hdia,rad,ir
        gc.collect()
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([urad,vtan,wr,qv,theta,heatsumr,hdiar,radr,irr],['u','v','w','qv','theta','heatsum','hdia','rad','ir'],self.window)
            del urad,vtan,qv,wr,heatsumr,hdiar,radr,irr
            gc.collect()
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
            print("---Finish!---")
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
            return outdict,r500
        else:
            uradC,vtanC,thetaC=da.from_array(self.swap_presaved('urad',True)[:,:,:,0:int(self.outsize)]),da.from_array(self.swap_presaved('vtan',True)[:,:,:,0:int(self.outsize)]),da.from_array(self.swap_presaved('theta',True)[:,:,:,0:int(self.outsize)])
            wC,qvC,hdiaC,rthratlwC,rthratswC,rthratlwcC,rthratswcC=self.read_azimuth_fields('W',False,True),\
            da.from_array(self.read_azimuth_fields('QVAPOR',False,True)[:,:,:,0:int(self.outsize)]),\
            self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True),self.read_azimuth_fields('RTHRATLWC',False,True),self.read_azimuth_fields('RTHRATSWC',False,True)
            heatsumC=hdiaC+rthratlwC+rthratswC
            radC = rthratlwC+rthratswC
            irC = rthratlwC+rthratswC-rthratlwcC-rthratswcC
            del rthratlwC,rthratswC,rthratlwcC,rthratswcC
            gc.collect()
            
            wCr = np.nan_to_num(wC[:,:,:,0:int(self.outsize)],nan=np.nanmean(wC[:,:,:,0:int(self.outsize)]))
            heatsumCr = np.nan_to_num(heatsumC[:,:,:,0:int(self.outsize)],nan=np.nanmean(heatsumC[:,:,:,0:int(self.outsize)]))
            hdiaCr = np.nan_to_num(hdiaC[:,:,:,0:int(self.outsize)],nan=np.nanmean(hdiaC[:,:,:,0:int(self.outsize)]))
            radCr = np.nan_to_num(radC[:,:,:,0:int(self.outsize)],nan=np.nanmean(radC[:,:,:,0:int(self.outsize)]))
            irCr = np.nan_to_num(irC[:,:,:,0:int(self.outsize)],nan=np.nanmean(irC[:,:,:,0:int(self.outsize)]))
            wCr,heatsumCr,hdiaCr,radCr,irCr = da.from_array(wCr),da.from_array(heatsumCr),da.from_array(hdiaCr),da.from_array(radCr),da.from_array(irCr)
            del wC,heatsumC,hdiaC,radC,irC
            gc.collect()
            
            uradL,vtanL,wL,qvL,thetaL,heatsumL,hdiaL,radL,irL=da.from_array(np.asarray(self.stickCTRL(uradC,urad,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(vtanC,vtan,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(wCr,wr,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(qvC,qv,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(thetaC,theta,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(heatsumCr,heatsumr,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(hdiaCr,hdiar,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(radCr,radr,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(irCr,irr,self.method)))
            del uradC,urad,theta,vtanC,thetaC,vtan,wCr,wr,qvC,qv,heatsumCr,heatsumr,hdiaCr,hdiar,radCr,radr,irCr,irr
            gc.collect()
                
            outdict=self.smooth_to_dict([uradL,vtanL,wL,qvL,thetaL,heatsumL,hdiaL,radL,irL],['u','v','w','qv','theta','heatsum','hdia','rad','ir'],self.window)
            del uradL,vtanL,wL,qvL,heatsumL,hdiaL,radL,irL
            gc.collect()
            print("---Finish!---")
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
            del outdict
            gc.collect()
            return None

    def preproc_onevar(self,varname='W',TYPEsmooth='orig',save_postfix=None):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
            
        if varname=='theta':
            qv=da.from_array(self.swap_presaved('theta',TYPE)[:,:,:,0:int(self.outsize)])
        else:
            qv=da.from_array(self.read_azimuth_fields(varname,False,True)[:,:,:,0:int(self.outsize)])
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([qv],[varname],self.window)
            del qv
            gc.collect()
            print("---Finish!---")
            return outdict
        else:
            if varname=='theta':
                qvC=da.from_array(self.swap_presaved('theta',True)[:,:,:,0:int(self.outsize)])
            else:
                qvC=da.from_array(self.read_azimuth_fields(varname,False,True)[:,:,:,0:int(self.outsize)])
            if TYPEsmooth=='orig':
                qvL=da.from_array(np.asarray(self.stickCTRL(qvC,qv,self.method)))
                del qvC,qv
                gc.collect()
                
                outdict=self.smooth_to_dict([qvL],[varname],self.window)
                del qvL
                gc.collect()
                print("---Finish!---")
                return outdict
            
    def preproc_dudvdw(self,smooth=True):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        
        urad,vtan=(self.swap_presaved('urad',TYPE)[:,:,:,0:int(self.outsize)]),(self.swap_presaved('vtan',TYPE)[:,:,:,0:int(self.outsize)])
        urad,vtan=da.from_array(np.nan_to_num(urad,nan=np.nanmean(urad))),da.from_array(np.nan_to_num(vtan,nan=np.nanmean(vtan)))
        theta=(self.swap_presaved('theta',TYPE)[:,:,:,0:int(self.outsize)])
        theta=da.from_array(np.nan_to_num(theta,nan=np.nanmean(theta)))
        w=(self.read_azimuth_fields('W',True,TYPE)[0][:,:,:,0:int(self.outsize)])
        wr = np.nan_to_num(w,nan=np.nanmean(w))
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([urad,vtan,w,theta],['u','v','w','theta'],self.window)
            durad,dvtan,dw,dtheta=self.forward_diff(arrayin=outdict['u'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['v'],delta=60*60,axis=0),\
            self.forward_diff(arrayin=outdict['w'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0)
            print(durad.shape)
        else:
            uradC,vtanC=(self.swap_presaved('urad',True)[:,:,:,0:int(self.outsize)]),(self.swap_presaved('vtan',True)[:,:,:,0:int(self.outsize)])
            uradC,vtanC=da.from_array(np.nan_to_num(uradC,nan=np.nanmean(uradC))),da.from_array(np.nan_to_num(vtanC,nan=np.nanmean(vtanC)))
            thetaC=(self.swap_presaved('theta',True)[:,:,:,0:int(self.outsize)])
            thetaC=da.from_array(np.nan_to_num(thetaC,nan=np.nanmean(thetaC)))
            wC=self.read_azimuth_fields('W',True,True)[0][:,:,:,0:int(self.outsize)]
            uradL,vtanL,wL,thetaL = da.from_array(np.asarray(self.stickCTRL(uradC,urad,self.method))),da.from_array(np.asarray(self.stickCTRL(vtanC,vtan,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(wC,w,self.method))),da.from_array(np.asarray(self.stickCTRL(thetaC,theta,self.method)))
            
            outdict=self.smooth_to_dict([uradL,vtanL,wL,thetaL],['u','v','w','theta'],self.window)
            durad,dvtan,dw,dtheta=self.forward_diff(arrayin=outdict['u'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['v'],delta=60*60,axis=0),\
            self.forward_diff(arrayin=outdict['w'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0)
            del uradC,vtanC,wC,urad,vtan,w,uradL,vtanL,wL,thetaL,theta,thetaC
            gc.collect()
        durad = np.nan_to_num(durad)
        dvtan = np.nan_to_num(dvtan)
        dw = np.nan_to_num(dw)
        dtheta = np.nan_to_num(dtheta)
        outdictuvw = {'du':durad,'dv':dvtan,'dw':dw,'dtheta':dtheta}
        
        print("---Finish!---")
        folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
        if smooth is True:
            read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdictuvw,'PICKLE')
        else:
            read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdictuvw,'PICKLE')                
        del outdict
        gc.collect()        
        return None
    
    def preproc_uvw(self):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        urad,vtan=da.from_array(self.swap_presaved('urad',TYPE)[:,:,:,0:int(self.outsize)]),da.from_array(self.swap_presaved('vtan',TYPE)[:,:,:,0:int(self.outsize)])
        w=da.from_array(self.read_azimuth_fields('W',True,TYPE)[0][:,:,:,0:int(self.outsize)])
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([urad,vtan,w],['u','v','w'],self.window)
        else:
            uradC,vtanC=da.from_array(self.swap_presaved('urad',True)[:,:,:,0:int(self.outsize)]),da.from_array(self.swap_presaved('vtan',True)[:,:,:,0:int(self.outsize)])
            wC=self.read_azimuth_fields('W',True,True)[0][:,:,:,0:int(self.outsize)]
            uradL,vtanL,wL = da.from_array(np.asarray(self.stickCTRL(uradC,urad,self.method))),da.from_array(np.asarray(self.stickCTRL(vtanC,vtan,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(wC,w,self.method)))
            
            outdict=self.smooth_to_dict([uradL,vtanL,wL],['u','v','w'],self.window)
            del uradC,vtanC,wC,urad,vtan,w,uradL,vtanL,wL
            gc.collect()
            
        outdictuvw = {'u':outdict['u'],'v':outdict['v'],'w':outdict['w']}
        print("---Finish!---")
        return outdictuvw
    
    def preproc_dthdQ(self):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        theta=da.from_array(self.swap_presaved('theta',TYPE)[:,:,:,0:int(self.outsize)])
        hdia,rthratlw,rthratsw=self.read_azimuth_fields('H_DIABATIC',False,TYPE),self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE)
        heatsum = hdia+rthratlw+rthratsw
        del hdia,rthratlw,rthratsw
        gc.collect()
        
        dQ = da.from_array(heatsum[:,:,:,0:int(self.outsize)])
        del heatsum
        gc.collect()
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([theta,dQ],['theta','heatsum'],self.window)
            dth,dQ=self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['heatsum'],delta=60*60,axis=0)
            print(dth.shape)
        else:
            thetaC=da.from_array(self.swap_presaved('theta',True)[:,:,:,0:int(self.outsize)])
            hdiaC,rthratlwC,rthratswC=self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True)
            heatsumC=hdiaC+rthratlwC+rthratswC
            del hdiaC,rthratlwC,rthratswC
            gc.collect()
            
            dQC = da.from_array(heatsumC[:,:,:,0:int(self.outsize)])
            del heatsumC
            gc.collect()
            
            thetaL,dQL = da.from_array(np.asarray(self.stickCTRL(thetaC,theta,self.method))),da.from_array(np.asarray(self.stickCTRL(dQC,dQ,self.method)))
            
            outdict=self.smooth_to_dict([thetaL,dQL],['theta','heatsum'],self.window)
            dth,dQ=self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['heatsum'],delta=60*60,axis=0)
            del thetaL,dQL
            gc.collect()
        dth = np.nan_to_num(dth)
        dQ = np.nan_to_num(dQ)
        outdictthQ = {'dth':dth,'dQ':dQ}
        print("---Finish!---")
        return outdictthQ
    
    def preproc_heateq(self,smooth=True):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
            
        hdia,rthratlw,rthratsw = self.read_azimuth_fields('H_DIABATIC',False,TYPE),self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE)
        rthratlwc,rthratswc = self.read_azimuth_fields('RTHRATLWC',False,TYPE),self.read_azimuth_fields('RTHRATSWC',False,TYPE)
        irlw,irsw = (rthratlw-rthratlwc),(rthratsw-rthratswc)
        
        rthratlwr = np.nan_to_num(rthratlw[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratlw[:,:,:,0:int(self.outsize)]))
        rthratswr = np.nan_to_num(rthratsw[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratsw[:,:,:,0:int(self.outsize)]))
        rthratlwcr = np.nan_to_num(rthratlwc[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratlwc[:,:,:,0:int(self.outsize)]))
        rthratswcr = np.nan_to_num(rthratswc[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratswc[:,:,:,0:int(self.outsize)]))
        irlwr = np.nan_to_num(irlw[:,:,:,0:int(self.outsize)],nan=np.nanmean(irlw[:,:,:,0:int(self.outsize)]))
        irswr = np.nan_to_num(irsw[:,:,:,0:int(self.outsize)],nan=np.nanmean(irsw[:,:,:,0:int(self.outsize)]))
        
        rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr = da.from_array(rthratlwr),da.from_array(rthratswr),da.from_array(rthratlwcr),da.from_array(rthratswcr),da.from_array(irlwr),da.from_array(irswr)
        del rthratlw,rthratsw,rthratlwc,rthratswc,irlw,irsw
        gc.collect()
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr],['LW','SW','LWC','SWC','IRLW','IRSW'],self.window)
            del rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr
            gc.collect()
            print("---Finish!---")
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix+'_heateq',outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix+'_heateq',outdict,'PICKLE')                
            del outdict
            gc.collect()
            return None
        else:
            hdiaC,rthratlwC,rthratswC=self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True)
            rthratlwcC,rthratswcC=self.read_azimuth_fields('RTHRATLWC',False,True),self.read_azimuth_fields('RTHRATSWC',False,True)
            irlwC,irswC = (rthratlwC-rthratlwcC),(rthratswC-rthratswcC)
                
            rthratlwrC = np.nan_to_num(rthratlwC[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratlwC[:,:,:,0:int(self.outsize)]))
            rthratswrC = np.nan_to_num(rthratswC[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratswC[:,:,:,0:int(self.outsize)]))
            rthratlwcrC = np.nan_to_num(rthratlwcC[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratlwcC[:,:,:,0:int(self.outsize)]))
            rthratswcrC = np.nan_to_num(rthratswcC[:,:,:,0:int(self.outsize)],nan=np.nanmean(rthratswcC[:,:,:,0:int(self.outsize)]))
            irlwrC = np.nan_to_num(irlwC[:,:,:,0:int(self.outsize)],nan=np.nanmean(irlwC[:,:,:,0:int(self.outsize)]))
            irswrC = np.nan_to_num(irswC[:,:,:,0:int(self.outsize)],nan=np.nanmean(irswC[:,:,:,0:int(self.outsize)]))    
            del hdiaC,rthratlwC,rthratswC,rthratlwcC,rthratswcC,irlwC,irswC
            gc.collect()
            
            rthratlwrL,rthratswrL = da.from_array(np.asarray(self.stickCTRL(rthratlwrC,rthratlwr,self.method))),da.from_array(np.asarray(self.stickCTRL(rthratswrC,rthratswr,self.method)))
            rthratlwcrL,rthratswcrL = da.from_array(np.asarray(self.stickCTRL(rthratlwcrC,rthratlwcr,self.method))),da.from_array(np.asarray(self.stickCTRL(rthratswcrC,rthratswcr,self.method)))
            irlwrL,irswrL = da.from_array(np.asarray(self.stickCTRL(irlwrC,irlwr,self.method))),da.from_array(np.asarray(self.stickCTRL(irswrC,irswr,self.method)))
            outdict=self.smooth_to_dict([rthratlwrL,rthratswrL,rthratlwcrL,rthratswcrL,irlwrL,irswrL],['LW','SW','LWC','SWC','IRLW','IRSW'],self.window)
            del rthratlwrC,rthratswrC,rthratlwcrC,rthratswcrC,irlwrC,irswrC,rthratlwr,rthratswr,rthratlwcr,rthratswcr,irlwr,irswr,rthratlwrL,rthratswrL,rthratlwcrL,rthratswcrL,irlwrL,irswrL
            gc.collect()
            print("---Finish!---")
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
            del outdict
            gc.collect()
            return None