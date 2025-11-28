import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys,gc
import numpy as np
import pickle
from tqdm.auto import tqdm
import random
        
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
#setup_seed(42)

class BranchEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BranchEncoder, self).__init__()

        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim1, latent_dim2, output_dim):
        super(Decoder, self).__init__()

        # Linear layers for both latent spaces to directly output to the regression output
        self.fc1 = nn.Linear(latent_dim1, output_dim, bias=True)
        self.fc2 = nn.Linear(latent_dim2, output_dim, bias=True)

    def forward(self, z1, z2):
        # Compute linear regression outputs from both latent spaces
        out1 = self.fc1(z1)
        out2 = self.fc2(z2)

        # Combine the outputs (sum them up)
        return out1 + out2


class VAE(nn.Module):
    def __init__(self, input_dim1, input_dim2, latent_dim1, latent_dim2, output_dim, brchindices):
        super(VAE, self).__init__()

        self.encoder1 = BranchEncoder(input_dim1, latent_dim1)
        self.encoder2 = BranchEncoder(input_dim2, latent_dim2)
        self.decoder = Decoder(latent_dim1, latent_dim2, output_dim)
        self.brchindices = brchindices

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        brchindex = list(np.asarray(self.brchindices).cumsum())#[0,50,38,50,8,50,20,20]).cumsum())
        X_lw, X_sw = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]]
        mu1, log_var1 = self.encoder1(X_lw)
        mu2, log_var2 = self.encoder2(X_sw)

        z1 = self.reparameterize(mu1, log_var1)
        z2 = self.reparameterize(mu2, log_var2)

        return self.decoder(z1, z2), mu1, log_var1, mu2, log_var2

def vae_loss(reconstructed_x, x, mu1, log_var1, mu2, log_var2, coeff):
    recon_loss = F.l1_loss(reconstructed_x, x, reduction='sum')
    kl_loss1 = -0.5 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp())
    kl_loss2 = -0.5 * torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp())
    return coeff*recon_loss + (1-coeff)*(kl_loss1 + kl_loss2), coeff*recon_loss, (1-coeff)*(kl_loss1 + kl_loss2)


def train_model(model=None,optimizer=None,scheduler=None,numepochs=None,early_stopper=None,variance_store=None,lossfunc=None,regularization='None',l1_lambda=0.01,l2_lambda=0.1,train_loader=None,val_loader=None,test_loader=None,count=None,
               vaeloss_coeff=1):
    # Custom loss: MSE_physicalLoss(eigenvectors,wcomps,variance_store)
    #liveloss = PlotLosses()
    schedulerCY,schedulerLS = scheduler[1],scheduler[0]
    train_losses,trainrecon_losses,trainkl_losses = [],[],[]
    val_losses,valrecon_losses,valkl_losses = [],[],[]
    val_NSEs = []
    statedicts = []
    for epoch in (range(int(numepochs))):
        """
        Initialize loss
        """
        train_loss = 0
        trainrecon_loss = 0
        trainkl_loss = 0
        """
        Operate per batch
        """
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            reconX,mu1,logvar1,mu2,logvar2 = model(features)
            batch_loss,recon_loss,kl_loss = vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,vaeloss_coeff)
            
            batch_loss.backward()                
            
            optimizer.step()
            schedulerCY.step()
            
            train_loss += batch_loss.item()
            trainrecon_loss += recon_loss.item()
            trainkl_loss += kl_loss.item()
            
            
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        trainrecon_loss = trainrecon_loss / len(train_loader)
        trainrecon_losses.append(trainrecon_loss)
        trainkl_loss = trainkl_loss / len(train_loader)
        trainkl_losses.append(trainkl_loss)
        
        model.train()
        criterion = vae_loss
        val_loss,valrecon_loss,valkl_loss = eval_model(model,
                                                       val_loader,
                                                       criterion,
                                                       l2_lambda,
                                                       vaeloss_coeff)
        schedulerLS.step(val_loss)
        statedicts.append(model.state_dict())
        
        ##################################################################
        # Early Stopping (valid / train)
        ##################################################################
        counter = 0
        if len(val_losses)>=1:
            best_score = val_losses[-1]
            if val_loss > best_score:
                counter += 1
                #val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
                valrecon_losses.append(valrecon_loss)
                valkl_losses.append(valkl_loss)
                if counter >= count:
                    break
            else:
                #val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
                valrecon_losses.append(valrecon_loss)
                valkl_losses.append(valkl_loss)
        else:
            #val_NSEs.append(val_NSE)
            val_losses.append(val_loss)
            valrecon_losses.append(valrecon_loss)
            valkl_losses.append(valkl_loss)
            
        if early_stopper:
            if early_stopper.__call__(val_loss, model):
                break
        
        if epoch % 300 == 0:
            print(((train_loss),(val_loss)))
            
    #return model, {'train':train_losses,'utrain':trainu_losses,'vtrain':trainv_losses,'wtrain':trainw_losses,'thtrain':trainth_losses,'val':val_losses} 
    return model, {'trainALL':train_losses,'valALL':val_losses,'trainRECON':trainrecon_losses,'valRECON':valrecon_losses,'trainKL':trainkl_losses,'valKL':valkl_losses}, statedicts

def eval_model(model, dataloader, loss_func, l2_lambda, vaeloss_coeff):
    with torch.no_grad():
        loss,loss2,loss3 = 0,0,0
        metric = 0
        
        global_sum = 0
        label_size = 0
        for feature, labels in dataloader:
            global_sum += labels.sum()
            label_size += len(labels)
            
        global_mean = global_sum / label_size
        model.train()
        for features, labels in dataloader:
            reconX,mu1,logvar1,mu2,logvar2 = model(features)
            batch_loss,recon_loss,kl_loss = vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,vaeloss_coeff)
            
            #l2_parameters = []
            #for parameter in model.parameters():
            #    l2_parameters.append(parameter.view(-1))
            #    l2 = l2_lambda * model.compute_l2_loss(torch.cat(l2_parameters))
            #batch_loss += l2
            loss+=batch_loss.item()
            loss2+=recon_loss.item()
            loss3+=kl_loss.item()
            
        num_batches = len(dataloader)
        
        loss = loss/num_batches
        loss2 = loss2/num_batches
        loss3 = loss3/num_batches
        return loss,loss2,loss3

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class resume_training:
    def __init__(self,splitnum=None,droprate=None,nonln_num=None,timelag=None,batch_size=None,num_workers=2,brchindex=None):
        self.splitnum=splitnum
        self.droprate=droprate
        self.vaeloss_coeff=nonln_num
        self.timelag = timelag
        self.batch_size = batch_size
        self.num_workers=2
        self.brchindex=brchindex
        
    def get_data(self,filepath=None):
        train_data,val_data,test_data = prepare_tensors(filepath,self.splitnum,self.timelag,'No')
        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=self.batch_size,shuffle=False)
        return train_loader,val_loader,test_loader
    
    def continue_training(self,datafilepath='./maria_store/',savefilepath='./maria_store/dropout_corr/',exp='e',scheduler_lr=[1e-14,5e-10]):
        train_loader,val_loader,_ = self.get_data(datafilepath)
        study = read_and_proc.depickle(savefilepath+str(self.splitnum)+'/bestparams.pkt')
        original_model = vae.VAE(self.brchindex[-2],self.brchindex[-1],1,1,1,self.brchindex)
        #else:
        #    study = read_and_proc.depickle(savefilepath+str(splitnum)+'/'+str(droprate)+'/'+'bestparams.pkt')
        #    original_model = ts_models.OptimMLR_lwsw_3D_ts_dropout2_nonln(self.droprate,self.brchindex,self.nonln_num)#[0,50,26,50,50,50,10,10],self.nonln_num)
        #######################################################################################################################################
        # Transfer state dict
        pretrained_model = torch.load(savefilepath+str(self.splitnum)+'/modelstest'+str(self.splitnum)+'_vae_exp1'+str(exp)+'.pk')[0]
        model_dict = original_model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        original_model.load_state_dict(model_dict)
        #######################################################################################################################################
        #######################################################################################################################################
        optimizer = torch.optim.Adam(original_model.parameters(), lr=study.best_params['lr'])
        #lossfunc = torch.nn.L1Loss()
        #scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-16, max_lr=5e-10,cycle_momentum=False) #1e-9/1e-5
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=scheduler_lr[0], max_lr=scheduler_lr[1],cycle_momentum=False) #1e-9/1e-5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-20)
        #######################################################################################################################################
        
        lowest_val_loss = float('inf')
        best_model = None
        schedulerCY,schedulerLS = scheduler2,scheduler
        train_losses,trainrecon_losses,trainkl_losses = [],[],[]
        val_losses,valrecon_losses,valkl_losses = [],[],[]
        
        for epoch in tqdm(range(20000)):
            original_model.train()
            train_loss = 0
            trainrecon_loss = 0
            trainkl_loss = 0
            # Training loop here
            for features, labels in train_loader:
                optimizer.zero_grad()
                reconX,mu1,logvar1,mu2,logvar2 = original_model(features)
                batch_loss,recon_loss,kl_loss = vae.vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,self.vaeloss_coeff)
                batch_loss.backward()
                optimizer.step()
                schedulerCY.step()
                
                train_loss += batch_loss.item() 
                trainrecon_loss += recon_loss.item()
                trainkl_loss += kl_loss.item()
                
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            trainrecon_loss = trainrecon_loss / len(train_loader)
            trainrecon_losses.append(trainrecon_loss)
            trainkl_loss = trainkl_loss / len(train_loader)
            trainkl_losses.append(trainkl_loss)

            # Validation loop
            original_model.eval()
            with torch.no_grad():
                val_loss = 0
                val_reconloss = 0
                val_klloss = 0
                val_loss,val_reconloss,val_klloss = 0,0,0
                for features, labels in val_loader:
                    reconX,mu1,logvar1,mu2,logvar2 = original_model(features)
                    batch_loss,recon_loss,kl_loss = vae.vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,self.vaeloss_coeff)
                    val_loss+=batch_loss.item()
                    val_reconloss+=recon_loss.item()
                    val_klloss+=kl_loss.item()
            
                val_loss = val_loss / len(val_loader)
                val_reconloss = val_reconloss / len(val_loader)
                val_klloss = val_klloss / len(val_loader)
                val_losses.append(val_loss)
                valrecon_losses.append(val_reconloss)
                valkl_losses.append(val_klloss)

            # Check if the current model has the lowest validation loss
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = original_model.state_dict()
                
            #torch.save(best_model, savefilepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/modelstest'+str(splitnum)+'_vae_'+str(times[i])+'.pk')
            torch.save(best_model, savefilepath+str(self.splitnum)+'/modelstest'+str(self.splitnum)+'_vae_exp1'+str(exp)+'_best.pk')
            read_and_proc.save_to_pickle(savefilepath+str(self.splitnum)+'/lossestest'+str(self.splitnum)+'_vae_exp1'+str(exp)+'_best.pkt',
                                         {'trainALL':train_losses,'valALL':val_losses,'trainRECON':trainrecon_losses,'valRECON':valrecon_losses,'trainKL':trainkl_losses,'valKL':valkl_losses},
                                         'PICKLE'
                                        )
        return None