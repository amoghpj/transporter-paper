import pandas as pd
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from utils import estimated_on_fraction, cmap1


strains = {'cd112':{'path':'1','desc':'HXT2','c':'k'},
           'hj101':{'path':'1','desc':'HXT2-sensorless','c':'r'}}
path = "./data/hxt2-wt-sensorless"
numrows = 8
numcols = 12
wom={}
counter = 1
for row in range(numrows):
    for col in range(1,numcols+1):
        if col % 2 !=0: #odd
            # For odd numbered wells, (col +1)/2 
            wom[counter] = row*numcols + int((col+1)/2)
        else:
            # For even wells, offset by numcols
            wom[counter] = row*numcols + int((col + numcols)/2)
        counter += 1
        
for strain, details in strains.items():
    print(strain)
    description=details['desc']
    expid = details['path']
    fig = plt.figure(figsize=(24,16))
    paramdf = pd.read_csv(f"{path}/extracted_params_{strain}-{expid}.csv",
                                      index_col=0).replace(np.nan,0.0)
    muThresh = paramdf['muThresh'].iloc[1]
    histdf = np.log10(pd.read_csv(f"{path}/plate_{strain}-{expid}.csv",
                                   index_col=0)/pd.read_csv(f"{path}/plate_{strain}-{expid}fsc.csv",
                                                            index_col=0))  + 4.
    wells = []
    dataset = []
    for row_ in reversed(range(0,8)):
        for col_ in range(0,12):
            wells.append(row_*12 + col_ + 1)
            
    for i,well in enumerate(wells):
        ax = fig.add_subplot(8,12,i+1)
        Y = histdf[str(well)].dropna().values
        y, bins = np.histogram(Y,bins=100)
        x = [b + (bins[1] - bins[0])/2 for b in bins[:-1]]
        ax.plot(x,y/sum(Y),'k',lw=5)
        ax.set_xlim(0,4.5)
        ax.set_ylim(0,0.07)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        params = paramdf.iloc[i]
        ax.set_facecolor(cmap1(estimated_on_fraction(x, params, muThresh)))
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()
    plt.savefig(f"{description}.pdf",dpi=200)
    plt.close()
    
fig = plt.figure(figsize=(24,16))
axlist = [fig.add_subplot(8,12,i) for i in range(1,97)]
for strain, details in strains.items():
    description=details['desc']
    expid = details['path']
    color = details['c']
    paramdf = pd.read_csv(f"{path}/extracted_params_{strain}-{expid}.csv",
                                      index_col=0).replace(np.nan,0.0)
    muThresh = paramdf['muThresh'].iloc[1]
    histdf = np.log10(pd.read_csv(f"{path}/plate_{strain}-{expid}.csv",
                                   index_col=0)/pd.read_csv(f"{path}/plate_{strain}-{expid}fsc.csv",
                                                            index_col=0))  + 4.
    wells = []
    dataset = []
    for row_ in reversed(range(0,8)):
        for col_ in range(0,12):
            wells.append(row_*12 + col_ + 1)
    for i,well in enumerate(wells):
        Y = histdf[str(well)].dropna().values
        y, bins = np.histogram(Y,bins=100)
        x = [b + (bins[1] - bins[0])/2 for b in bins[:-1]]
        axlist[i].plot(x,y/sum(Y),color,lw=5)
        axlist[i].set_xlim(0,4.5)
        axlist[i].set_ylim(0,0.07)
        axlist[i].set_yticklabels([])
        axlist[i].set_xticklabels([])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"comparison.pdf",dpi=200)
plt.close()
