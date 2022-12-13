import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import estimated_on_fraction, cmap1

root = './data/maltose-galactose-dg/'
platedata = {'high':{'plates':['cenpk-mal-gal-high-16-1',
                               'cenpk-mal-gal-high-16-2'],
                     'ncols':8,
                     'outname':'fig2-cleaned'},
             'low':{'plates':['cenpk-mal-gal-low-24-1',
                              'cenpk-mal-gal-low-24-2'],
                    'ncols':7,
                    'outname':'figs1-cleaned'}}

print("Plotting...")
for platetype, details in platedata.items():
    print(f"{platetype}")
    fig = plt.figure(figsize=(16,24))
    nrows = 12
    ncols = details['ncols']
    axes = [fig.add_subplot(nrows,ncols,i) for i in range(1,nrows*ncols+1)]
    layout = np.array([i for i in range(1,97)]).reshape(8,12).T
    layout = np.flipud(layout[:,8-ncols:])
    dflist = [np.log10(pd.read_csv(f"{root}{p}.csv",
                        index_col=0)/pd.read_csv(f"{root}{p}_fsc.csv",
                        index_col=0)) + 4 for p in details['plates']]
    df = pd.concat(dflist)

    paramsdflist = pd.concat([pd.read_csv(f'{root}extracted_params_{p}.csv',index_col='id') for p in details['plates']])
    paramsdf = paramsdflist.groupby(paramsdflist.index).mean()
    data = df[str(1)].dropna().values
    _, bins = np.histogram(data,bins=100)
    xwin = (bins[1] - bins[0])/2
    x = [b + xwin for b in bins[:-1]]
    muThresh = paramsdf.muThresh.iloc[0]
    counter = 0
    extracted_params = paramsdf.apply(estimated_on_fraction, axis='columns',args=(x,muThresh)).values
    extracted_params = np.fliplr(np.flipud(extracted_params.reshape(8,12).T))[:,8-ncols:]
    for i in range(12):
        for j  in range(ncols):
            well = layout[i,j]
            wellc = extracted_params[i,j]
            data = df[str(well)].dropna().values
            y, bins = np.histogram(data,bins=100)
            y = y/y.sum()
            xwin = (bins[1] - bins[0])/2
            x = [b + xwin for b in bins[:-1]]
            axes[counter].plot(x,y,'k',lw=3)
            axes[counter].set_facecolor(cmap1(wellc))
            axes[counter].axvline(muThresh)
            axes[counter].text(3,0.05,str(round(wellc,1)))
            counter += 1

    for ax in axes:
        ax.set_xlim(0,5)
        ax.set_ylim(0,0.07)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{details["outname"]}.pdf',dpi=300,bbox_inches="tight")        
    plt.close()
        
    
