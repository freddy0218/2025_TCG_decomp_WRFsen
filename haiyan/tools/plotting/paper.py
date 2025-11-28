import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import glob,os,sys
from tqdm.auto import tqdm
import proplot as plot
import pandas as pd

plot.rc.update({'figure.facecolor':'w','axes.labelweight':'ultralight',
                'tick.labelweight':'ultralight','gridminor.linestyle':'--','title.weight':'normal','linewidth':0.5})
plot.rc.metacolor = 'k'
plot.rc.update({'axes.labelweight':'normal','tick.labelweight':'normal','figure.facecolor':'w','title.color':'k','fontname': 'Source Sans Pro', 'fontsize': 11,'linewidth':1.25})
plot.rc.axesfacecolor = 'w'

def plot_nonradstats_overview(indata=None,toplots=None,reshapesize=[10,208],yaxis=None,nonradsource=None,nonradnum=None,ex_var_ratio=None,suptitle='Haiyan 2D (6hr)',saveloc='../figures/1010/fig8_haiyan6hr_corr.png'):
    """
    indata: FI
    toplots: eigenvectors
    reshapesize: 1D->multi-d
    nonradsource,nonradnum: first 5 non-RAD
    """
    fig, axs = plot.subplots([[1, 1, 1, 1,1], [2,3,4,5,6]], refnum=2, refwidth=1.57,refheight=1.25,wspace=(0, 0, 0, 0),height_ratios=[1.75, 1],sharey=0)
    axs[0].boxplot(indata, means=True, marker='x', meancolor='r', fillcolor='gray4', orientation='horizontal',showfliers=False)
    axs[0].format(xlabel=r'Feature Importance (*$\mathit {10^4}$)',ylabel='non-RAD PCs',title=r'non-RAD PCs Feature Importance (*$\mathit {10^4}$)')#xlim=[0,0.0002*1.5e4])#ylim=[0,20])
    titles = [str(nonradsource[i])+str(nonradnum[i]+1) for i in range(5)]
    RAD=False
    for inx,i in enumerate(toplots):
        tempdata = toplots[inx].reshape(reshapesize[0],reshapesize[1])
        pcw=axs[inx+1].contourf(np.linspace(0,207,208)*3,yaxis,(tempdata-np.nanmean(tempdata))/np.nanstd(tempdata),cmap='balance',levels=np.linspace(-1.5,1.5,31),extend='both')
        if RAD is True:
            axs[inx+1].format(title=f'RAD PC#{str(inx+1)} ({ex_var_ratio[inx]}%)')
        else:
            axs[inx+1].format(title=titles[inx]+' '+f'({ex_var_ratio[inx]}%)')
            axs[inx+1].format(xlabel=r'Distance from TC Centre (km)',ylabel='')
    axs[1].format(ylabel='Pressure (hPa)')
    for i in range(2,6):
        axs[i].format(yticklabels=[])
    fig.colorbar(pcw, loc='b',ticks=np.linspace(-1.5,1.5,31)[-1]/3)
    axs.format(suptitle=suptitle)
    plt.savefig(saveloc,dpi=300)
    plt.show()
    return None