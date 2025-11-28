from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import cross_val_score
import numpy as np
import gc
import importlib
import sys
from tqdm.auto import tqdm
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr,proc_mlrfcst,maria_IO
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan

def delete_padding(inTS=None,outTS=None):
    output_nozero,input_nozero = [],[]
    for i in range(len(outTS[:,0])):
        temp = outTS[i,:]
        tempin = inTS[i,:]
        if temp.all()==0:
            continue
        else:
            output_nozero.append(temp)
            input_nozero.append(tempin)
    return input_nozero,output_nozero

class forwardfeatureadder(BaseEstimator,SelectorMixin,MetaEstimatorMixin):
    """Transformer to add feature at a sequential order
    Parameters:
    estimator: Regression model
    n_features_to_select: number of features to add to the model
    cv: how many folds would we want during cross-validation
    n_jobs: Parallelization
    startfeatures: Features we would like to include in the model without cross-validation [we do this to accentuate the role of heating]
    
    Output:
    self instance
    """
    def __init__(self,estimator,n_features_to_select=None,cv=5,n_jobs=None,startfeatures=None,PCAdict=None,Afdict=None,numcomp=None,LT=None,optigoal='surface',Xsurf=None,Ysurf=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.n_jobs = n_jobs
        self.startfeatures = startfeatures
        self.PCAdict = PCAdict
        self.Afdict = Afdict
        self.numcomp = numcomp
        self.LT = LT
        self.optigoal = optigoal
        self.Xsurf=Xsurf
        self.Ysurf=Ysurf
    
    def get_real_winds(self):
        temp1,temp2,temp3,temp4 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,
                            forecastPC=None).windrates_real(LT=self.LT)
        return temp1,temp2,temp3,temp4
    
    def convert_forecast_winds(self,yforecast=None):
        try:
            temp1,temp2,temp3,temp4 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,
                                                                      forecastPC=yforecast).output_reshapeRECON(forecast_eig=yforecast[int(self.LT[i]-1)])
            return temp1,temp2,temp3,temp4
        except:
            temp1,temp2 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,
                                                                      forecastPC=yforecast,target='surface').output_reshapeRECON(forecast_eig=yforecast)        
            return temp1,temp2
    
    def fit(self, X,y=None):
        """Learn features to select from X.
        X (n_samples,n_features): Training vectors
        Y (n_samples): Target values
        """
        # Define basic settings
        n_features = X.shape[1]
        current_mask = np.zeros(shape=n_features,dtype=bool)
        for index in self.startfeatures:
            current_mask[index] = True
        n_iteractions = self.n_features_to_select
        
        # Do forward selection
        addinput,r2 = [],[]
        clone_estimator = clone(self.estimator)
        for _ in range(n_iteractions):
            #new_feature_idx,r2t = self.get_best_new_feature_R2based(clone_estimator,X,y,current_mask)
            new_feature_idx = self.get_best_new_feature(clone_estimator,X,y,current_mask)
            #r2.append(r2t)
            current_mask[new_feature_idx] = True
            addinput.append(current_mask)
        
        self.support_ = current_mask
        self.new_feature = new_feature_idx
        self.r2 = r2
        return self
    
    def get_best_new_feature(self,estimator,X,y,current_mask):
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores={}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            
            # Add a new feature
            X_new = X[:,candidate_mask]
            # Improvement
            scores[feature_idx] = cross_val_score(estimator,X_new,y,cv=self.cv,scoring=None,n_jobs=self.n_jobs).mean()
        return max(scores,key=lambda feature_idx: scores[feature_idx])
    
    #--------------------------------------------------------------
    # To do -> Add featureselector based on r2
    # Candidate mask -> Xnew
    # fit linear model with (Xnew,y)
    # {output r2 term [time consideration => target: surface u/v]}...repeat for all u/v/w/theta members
    # get component index that results in best r2 score
    # --------[[Exit loops when r2 reaches 0.75?]]-----------------
    # add to mask during fitting 
    #---------------------------------------------------------------------------------------------------
    def get_best_new_feature_R2based(self,estimator,X,y,current_mask):             
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores={}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            # Add a new feature
            X_new = X[:,candidate_mask]
            # Improvement
            LDTME = np.linspace(0,44,45)+1
            y_forecast = mlr.model_fitpredict(X_new,y,estimator,LDTME).modelfit(singleLT=True)[0].predict(X_new)
            #################################################################################################################################################################################################################
            # Forecast winds
            #################################################################################################################################################################################################################
            if self.optigoal=='surface':
                teMP1,teMP2 = self.convert_forecast_winds(y_forecast)
                teMP1s,teMP2s = (teMP1.reshape(teMP1.shape[0],39,360,167)[:,0,:,:]).reshape(teMP1.shape[0],360*167),(teMP2.reshape(teMP2.shape[0],39,360,167)[:,0,:,:]).reshape(teMP2.shape[0],360*167)
                del teMP1,teMP2
                gc.collect()
                scores[feature_idx] = r2_score(np.concatenate((self.Xsurf,self.Ysurf),axis=0),np.concatenate((teMP1s,teMP2s),axis=0))
            else:
                teMP1,teMP2,teMP3,teMP4 = self.convert_forecast_winds(y_forecast)                
                scores[feature_idx] = r2_score(np.concatenate((reteMP1,reteMP2,reteMP3,reteMP4),axis=0),np.concatenate((teMP1,teMP2,teMP3,teMP4),axis=0))
        return max(scores,key=lambda feature_idx: scores[feature_idx]),max(scores.values())
    
    def _get_support_mask(self):
        return self.support_
    
class random_FFT:
    def __init__(self,validindex=None,testindex=None,varnum=None,LT=None):
        self.validindex = validindex
        self.testindex = testindex
        self.varnum = varnum
        self.LT = LT
        
    def output_realdterms(self,category='train',varnum=None,LT=24):
        if self.varnum==2:
            a,b,_,_ = retrieve_cartesian(PCA_dict=None,Af_dict=haiyan_data,numcomp=None,LT=self.LT,forecastPC=None,\
                                         target='all',suffix=suffix).windrates_real(uvwheatpath='TCGphy/testML/output/haiyan/processed/uvwheat/',\
                                                                                    LT=LT,category=category,validindex=self.validindex,testindex=self.testindex)
            return a,b
        elif self.varnum==4:
            a,b,c,d = retrieve_cartesian(PCA_dict=None,Af_dict=haiyan_data,numcomp=None,LT=self.LT,forecastPC=None,\
                                         target='all',suffix=suffix).windrates_real(uvwheatpath='TCGphy/testML/output/haiyan/processed/uvwheat/',\
                                                                                    LT=LT,category=category,validindex=self.validindex,testindex=self.testindex)     
            return a,b,c,d
        
    def separate_realwinds(self,vertshape=10):
        realdu,realdv,realdw,realdtheta = self.output_realdterms(category='train',varnum=self.varnum,LT=self.LT)
        realdu_valid,realdv_valid,realdw_valid,realdtheta_valid = self.output_realdterms(category='valid',varnum=self.varnum,LT=self.LT)
        realdu_test,realdv_test,realdw_test,realdtheta_test = self.output_realdterms(category='test',varnum=self.varnum,LT=self.LT)
        ################################################################################################################################################
        # Surface winds
        ################################################################################################################################################
        radsize = int(realdu.shape[1]/vertshape/360)
        realsurfu,realsurfv = realdu.reshape(realdu.shape[0],vertshape,360,radsize)[:,0,:,:].reshape(realdu.shape[0],360*radsize),realdv.reshape(realdv.shape[0],vertshape,360,radsize)[:,0,:,:].reshape(realdu.shape[0],360*radsize)
        realsurfu_valid,realsurfv_valid = realdu_valid.reshape(realdu_valid.shape[0],vertshape,360,radsize)[:,0,:,:].reshape(realdu_valid.shape[0],360*radsize),realdv_valid.reshape(realdv_valid.shape[0],vertshape,360,radsize)[:,0,:,:].reshape(realdu_valid.shape[0],360*radsize)
        realsurfu_test,realsurfv_test = realdu_test.reshape(realdu_test.shape[0],vertshape,360,radsize)[:,0,:,:].reshape(realdu_test.shape[0],360*radsize),realdv_test.reshape(realdv_test.shape[0],vertshape,360,radsize)[:,0,:,:].reshape(realdu_test.shape[0],360*radsize)
        del realdu,realdv,realdw,realdtheta,realdu_valid,realdv_valid,realdw_valid,realdtheta_valid,realdu_test,realdv_test,realdw_test,realdtheta_test
        gc.collect()
        return {'u':realsurfu,'v':realsurfv},{'u':realsurfu_valid,'v':realsurfv_valid},{'u':realsurfu_test,'v':realsurfv_test}
    
    def separate_realall(self,vertshape=10):
        realdu,realdv,realdw,realdtheta = self.output_realdterms(category='train',varnum=self.varnum,LT=self.LT)
        realdu_valid,realdv_valid,realdw_valid,realdtheta_valid = self.output_realdterms(category='valid',varnum=self.varnum,LT=self.LT)
        realdu_test,realdv_test,realdw_test,realdtheta_test = self.output_realdterms(category='test',varnum=self.varnum,LT=self.LT)
        return {'u':realdu,'v':realdv,'w':realdw,'theta':realdtheta},{'u':realdu_valid,'v':realdv_valid,'w':realdw_valid,'theta':realdtheta_valid},{'u':realdu_test,'v':realdv_test,'w':realdw_test,'theta':realdtheta_test}
    
    def preproc_INOUT(self,yall=None,Xtrain=None,expTYPE='dtthuvw',nummem=[50,38,91,8],optimizeto='train'):
        # Initiate model [54,26,50,7
        linreg = LinearRegression()
        if optimizeto=='train':
            ytrain = [yobj[0] for yobj in yall]
        elif optimizeto=='valid':
            ytrain = [yobj[1] for yobj in yall]
        mlrIN,mlrOUT = mlr.SimpleIOhandler(LT=self.LT,auxIN=None).transform(Xtrain[expTYPE],ytrain)
        return mlrIN,mlrOUT,linreg
    
    def preproc_INOUT_maria(self,yall=None,Xtrain=None,nummem=[50,38,91,8],optimizeto='train'):
        # Initiate model [54,26,50,7
        linreg = LinearRegression()
        if optimizeto=='train':
            ytrain = yall[0]#[yobj[0] for yobj in yall]
        elif optimizeto=='valid':
            ytrain = yall[1]#[yobj[1] for yobj in yall]
        mlrIN,mlrOUT = delete_padding(Xtrain,ytrain)#mlr.SimpleIOhandler(LT=self.LT,auxIN=None).transform(Xtrain[expTYPE],ytrain)
        return mlrIN,mlrOUT,linreg
    
    def do_seq_FS(self,data=None,PCAdict=None,yall=None,Xtrain=None,Xvalid=None,expTYPE='dtthuvw',\
                  findextra=40,cv=7,nummem=[50,38,91,8],holdmem=None,optimizeto='train'):
        model,reducedX,new_feature = [],[],[]
        if optimizeto=='train':
            mlrIN,mlrOUT,linreg = self.preproc_INOUT(yall,Xtrain,expTYPE,nummem,'train')
        elif optimizeto=='valid':
            mlrIN,mlrOUT,linreg = self.preproc_INOUT(yall,Xvalid,expTYPE,nummem,'valid')
            mlrIN_train,mlrOUT_train,_ = self.preproc_INOUT(yall,Xtrain,expTYPE,nummem,'train')
        #holdmem = holdmem
        for i in (range(findextra)):#np.asarray(mlrIN).shape[1]-20)):
            seq_temp = forwardfeatureadder(linreg,1,cv,4,holdmem,\
                                           PCAdict,data,nummem,self.LT,'surface',None,None).fit(np.asarray(mlrIN),mlrOUT)
            holdmem.append(seq_temp.new_feature)
            new_feature.append(seq_temp.new_feature)
            if optimizeto=='train':
                Xn = seq_temp.transform(np.asarray(mlrIN)) #Remove unimportant features
                ridge_reduced = LinearRegression().fit(Xn,mlrOUT) #Train model with reduced input
            elif optimizeto=='valid':
                Xn = seq_temp.transform(np.asarray(mlrIN_train))
                ridge_reduced = LinearRegression().fit(Xn,mlrOUT_train)
            model.append(ridge_reduced)
            reducedX.append(Xn)
        return model,reducedX,new_feature
    
    def do_seq_FS_maria(self,data=None,PCAdict=None,yall=None,Xtrain=None,Xvalid=None,\
                  findextra=40,cv=7,nummem=[50,38,91,8],holdmem=None,optimizeto='train'):
        model,reducedX,new_feature = [],[],[]
        if optimizeto=='train':
            mlrIN,mlrOUT,linreg = self.preproc_INOUT_maria(yall,Xtrain,nummem,'train')
        elif optimizeto=='valid':
            mlrIN,mlrOUT,linreg = self.preproc_INOUT_maria(yall,Xvalid,nummem,'valid')
            mlrIN_train,mlrOUT_train,_ = self.preproc_INOUT_maria(yall,Xtrain,nummem,'train')
        #holdmem = holdmem
        for i in tqdm(range(findextra)):#np.asarray(mlrIN).shape[1]-20)):
            seq_temp = forwardfeatureadder(linreg,1,cv,4,holdmem,\
                                           PCAdict,data,nummem,self.LT,'surface',None,None).fit(np.asarray(mlrIN),mlrOUT)
            holdmem.append(seq_temp.new_feature)
            new_feature.append(seq_temp.new_feature)
            if optimizeto=='train':
                Xn = seq_temp.transform(np.asarray(mlrIN)) #Remove unimportant features
                ridge_reduced = LinearRegression().fit(Xn,mlrOUT) #Train model with reduced input
            elif optimizeto=='valid':
                Xn = seq_temp.transform(np.asarray(mlrIN_train))
                ridge_reduced = LinearRegression().fit(Xn,mlrOUT_train)
            model.append(ridge_reduced)
            reducedX.append(Xn)
        return model,reducedX,new_feature