import numpy as np
import torch
import properscoring as ps
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random,sys,gc
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/2024_TCG_VED_WRFsen/')
from util.ml import (preproc,vae)
from util.wrf_process import (read_and_write)

class ved_analysis:
    def __init__(self,splitnum=None,vaeloss_coeff=None,batch_size=None,num_workers=2,PCApaths='storage/proc/PCA/PCAsmooth9103*',Xpaths='storage/proc/Xsmooth/9103/Xtimeseries*',
                VEDpaths='storage/proc/VEDsmooth_9103/',startname=22):
        self.splitnum=splitnum
        self.vaeloss_coeff=vaeloss_coeff
        self.batch_size = batch_size
        self.num_workers=2
        self.PCApaths=PCApaths
        self.Xpaths=Xpaths
        self.VEDpaths=VEDpaths
        self.startname=int(startname)
        
    def get_data(self,suffix,config_set):
        PCA = read_and_write.depickle(sorted(glob.glob(suffix+self.PCApaths))[self.splitnum])
        X = read_and_write.depickle(sorted(glob.glob(suffix+self.Xpaths))[self.splitnum])
        y = read_and_write.depickle(sorted(glob.glob(suffix+'storage/proc/y*'))[self.splitnum])
        try:
            X['test'] = X.pop('Xtest')
        except KeyError:
            pass
        
        validindices = sorted(glob.glob(suffix+self.Xpaths))[self.splitnum].split('/')[-1][self.startname:].split('.')[0]
        RADstops = read_and_write.depickle(glob.glob(suffix+self.VEDpaths+str(validindices)+f'/losscoeff_{str(self.vaeloss_coeff)}/nummem.pkl')[0])
        #LWstop = np.abs(PCA['PCA']['LW'].explained_variance_ratio_.cumsum()-float(config_set['ML_LWnumcomps'])).argmin()
        #SWstop = np.abs(PCA['PCA']['SW'].explained_variance_ratio_.cumsum()-float(config_set['ML_SWnumcomps'])).argmin()
        train_data,val_data,test_data = preproc.prepare_tensors(X,y,'No')
        NTdata = preproc.prepare_tensors(X,y,'Yes')
        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=self.batch_size,shuffle=False)
        return train_loader,val_loader,test_loader,RADstops,NTdata,validindices
        
    def load_retrain_and_update(self,retrain,orig):
        # Read in original model
        orig_model = modelorig[0]
        # Get the weights
        orig_dict = orig_model.state_dict()
        # Update weights
        new_dict = {k: v for k, v in retrain.items() if k in orig_dict}
        orig_dict.update(new_dict)
        # Load updated weights
        orig_model.load_state_dict(orig_dict)
        return orig_model
        
    def grab_predictions(self,model=None,Xtensors=None,trial=20,TYPE='ved'):
        if TYPE=='ved':
            predds = [np.squeeze(model.train()(Xtensors)[0].detach().numpy().transpose()) for i in range(trial)]
            return predds,np.nanmean(np.asarray(predds),axis=0)
        elif TYPE=='Drop1_2':
            model.dropout1.train()
            model.dropout2.train()
            predds = [np.squeeze(model.train()(Xtensors)[0].detach().numpy().transpose()) for i in range(trial)]
            return predds,np.nanmean(np.asarray(predds),axis=0),storeweights
            
    def get_scores(self,allpred,meanpred,truth):
        meanr2 = get_meanr2(np.asarray(meanpred).transpose(),np.asarray(truth).transpose())
        meanrmse = get_meanrmse(np.asarray(meanpred).transpose(),np.asarray(truth).transpose())
        meanmae = get_meanmae(np.asarray(meanpred).transpose(),np.asarray(truth).transpose())
        crps = ps.crps_ensemble(np.asarray(truth).transpose(),np.asarray(allpred).transpose()).mean()
        spread = np.std(np.asarray(allpred),axis=0)
        return {'r2':meanr2,'rmse':meanrmse,'mae':meanmae,'crps':crps,'spread':spread}
        
    def inner_loop_scores(self,suffix,config_set,repeatnum):
        """
        splitnum; ved_losscoeff; suffix; config_set; repeatnum (Times to repeat the probabilistic forecast)
        """
        train_data,val_data,test_data,numcomps,NT_data,_ = self.get_data(suffix,config_set)
        alldicts = []
        for exp in ['exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']:
            modelsss = torch.load(suffix+self.VEDpaths+str(sorted(glob.glob('../../'+self.Xpaths))[self.splitnum].split('/')[-1][self.startname:].split('.')[0])+
                                  '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'modelstest_vae_'+str(exp)+'_best.pk')
            modelorig = torch.load(suffix+self.VEDpaths+str(sorted(glob.glob('../../'+self.Xpaths))[self.splitnum].split('/')[-1][self.startname:].split('.')[0])+
                                   '/losscoeff_'+str(self.vaeloss_coeff)+'/'+'modelstest_vae_'+str(exp)+'.pk')
            updated_model = modelsss#self.load_retrain_and_update(modelsss,modelorig)
            params,names = model_outweights(updated_model)

            alltrain,meantrain = self.grab_predictions(updated_model,torch.FloatTensor(NT_data['train'][0]).to('cpu'),repeatnum,'ved')
            allvalid,meanvalid = self.grab_predictions(updated_model,torch.FloatTensor(NT_data['valid'][0]).to('cpu'),repeatnum,'ved')
            alltest,meantest = self.grab_predictions(updated_model,torch.FloatTensor(NT_data['test'][0]).to('cpu'),repeatnum,'ved')

            train_performance = self.get_scores(alltrain,meantrain,NT_data['train'][1])
            valid_performance = self.get_scores(allvalid,meanvalid,NT_data['valid'][1])
            test_performance = self.get_scores(alltest,meantest,NT_data['test'][1])

            tosave_train = {'all':alltrain,'mean':meantrain,'scores':train_performance,'truth':NT_data['train'][1]}
            tosave_valid = {'all':allvalid,'mean':meanvalid,'scores':valid_performance,'truth':NT_data['valid'][1]}
            tosave_test = {'all':alltest,'mean':meantest,'scores':test_performance,'truth':NT_data['test'][1]}
            alldicts.append({'train':tosave_train,'valid':tosave_valid,'test':tosave_test,'model':modelsss})
        return alldicts
        
def find_best_model_loc(alldicts_seeds,TYPE,scoreTYPE):
    allscore = []
    for i in range(len(alldicts_seeds)):
        temp = alldicts_seeds[i]
        score = []
        for j in ([0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]):
            temp2 = temp[j]
            score.append([obj[TYPE]['scores'][scoreTYPE] for obj in temp2])
        allscore.append(score)
        
    scoremean = []
    for i in range(len(alldicts_seeds)):
        scoremean.append([np.asarray(obj).mean() for obj in allscore[i]])

    if scoreTYPE=='r2':
        bestsplit_loc = np.asarray([np.asarray(obj).max() for obj in scoremean]).argmax()
        bestcoeff_loc = np.asarray([np.asarray(obj).mean() for obj in (allscore[bestsplit_loc])]).argmax()
        bestmodel_loc = np.asarray(allscore[bestsplit_loc][bestcoeff_loc]).argmax()
        bestscore     = np.asarray(allscore[bestsplit_loc][bestcoeff_loc]).max()
    else:
        bestsplit_loc = np.asarray([np.asarray(obj).min() for obj in scoremean]).argmin()
        bestcoeff_loc = np.asarray([np.asarray(obj).mean() for obj in (allscore[bestsplit_loc])]).argmin()
        bestmodel_loc = np.asarray(allscore[bestsplit_loc][bestcoeff_loc]).argmin()
        bestscore     = np.asarray(allscore[bestsplit_loc][bestcoeff_loc]).min()
    return bestsplit_loc,bestcoeff_loc,bestmodel_loc,bestscore

def output_1score(alldicts_seeds,TYPE,scoreTYPE):
    allscore = []
    for i in range(len(alldicts_seeds)):
        temp = alldicts_seeds[i]
        score = []
        for j in ([0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]):
            temp2 = temp[j]
            score.append([obj[TYPE]['scores'][scoreTYPE] for obj in temp2])
        allscore.append(score)
    return allscore


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def model_outweights_all(model=None):
    params,names = [],[]
    for name, param in model.named_parameters():
        params.append(param)
        names.append(name)
    return params, names

def model_outweights(model=None):
    params,names = [],[]
    for name, param in model.named_parameters():
        if ".weight" not in name:
            continue            
        else:
            params.append(param)
            names.append(name)
    return params, names

class analysis_patterns:
    def __init__(self,bestdropout_index=7,dropout_rates=None):
        self.bestdropout_index=bestdropout_index
        self.dropout_rates=dropout_rates
    
    def new_structure_vae(self,X_totrain=None,bestmodel=None,TYPE='LW',varINDX=[-40,-20]):
        LW,SW = (np.asarray(X_totrain)[:,varINDX[0]:varINDX[1]]),(np.asarray(X_totrain)[:,varINDX[1]:])
        store = []
        if TYPE=='LW':
            for i in range(np.abs(varINDX[1])):
                term1 = model_outweights_all(bestmodel)[0][0][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][0][0].detach().numpy()/np.std(LW,axis=0))**2))
                term2 = (np.std(LW[:,i])*sumss)
                store.append(np.sign(model_outweights_all(bestmodel)[0][-4][0].detach().numpy()[0])*term1/term2)
        elif TYPE=='SW':
            for i in range(np.abs(varINDX[2]-varINDX[1])):
                term1 = model_outweights_all(bestmodel)[0][4][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][4][0].detach().numpy()/np.std(SW,axis=0))**2))
                term2 = (np.std(SW[:,i])*sumss)
                store.append(np.sign(model_outweights_all(bestmodel)[0][-2][0].detach().numpy()[0])*term1/term2)
        elif TYPE=='LW_logvar':
            for i in range(np.abs(varINDX[1])):
                term1 = model_outweights_all(bestmodel)[0][2][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][2][0].detach().numpy()/np.std(LW,axis=0))**2))
                term2 = (np.std(LW[:,i])*sumss)
                store.append(term1/term2)  
        elif TYPE=='SW_logvar':
            for i in range(np.abs(varINDX[2]-varINDX[1])):
                term1 = model_outweights_all(bestmodel)[0][6][0].detach().numpy()[i]
                sumss = np.sqrt(np.sum((model_outweights_all(bestmodel)[0][6][0].detach().numpy()/np.std(SW,axis=0))**2))
                term2 = (np.std(SW[:,i])*sumss)
                store.append(term1/term2)
        return store
        
def get_meanr2(X=None,y=None):
    return r2_score(y,X)
    
def get_meanrmse(X=None,y=None):
    return np.sqrt(mean_squared_error(y,X))
    
def get_meanmae(X=None,y=None):
    return mean_absolute_error(y,X)