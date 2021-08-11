import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from scipy.stats import norm
import sys
cmap_name = "viridis"
def estimated_on_fraction(x, params):
    estimated_of = 0.0
    on_level = params['on_level']
    off_level = params['off_level']
    on_fraction = params['on_fraction']
    off_fraction = params['off_fraction']
    on_sigma = params['on_sigma']
    off_sigma = params['off_sigma']
    estimated_of = 0
    if off_sigma > 0:
        yfit_off = off_fraction*norm.pdf(x, loc=off_level,
                                        scale=np.sqrt(off_sigma))
        denom = sum(yfit_off)
        num = sum([off_fraction*norm.pdf(_x, loc=off_level,
                                              scale=np.sqrt(off_sigma))
                   for _x in x if _x > muThresh])            
        estimated_of += num/denom*off_fraction
    if on_sigma > 0:
        yfit_on = on_fraction*norm.pdf(x, loc=on_level,
                                        scale=np.sqrt(on_sigma))            

        denom = sum(yfit_on)
        num = sum([on_fraction*norm.pdf(_x, loc=on_level,
                                              scale=np.sqrt(on_sigma))
                   for _x in x if _x > muThresh])
        estimated_of += num/denom*on_fraction
    return(estimated_of)

root = './data/maltose-galactose-dg/'
platedata = {# 'high':{'plates':['cenpk-mal-gal-high-16-1',
             #                   'cenpk-mal-gal-high-16-2'],
             #         'ncols':8,
             #         'outname':'fig2'},
             'low':{'plates':['cenpk-mal-gal-low-24-1',
                              'cenpk-mal-gal-low-24-2'],
                    'ncols':7,
                    'outname':'figs1'}}
for platetype, details in platedata.items():
    fig = plt.figure(figsize=(16,24))
    nrows = 12
    ncols = details['ncols']
    axes = [fig.add_subplot(nrows,ncols,i) for i in range(1,nrows*ncols+1)]

    cmap1 = matplotlib.colors.ListedColormap(cm.get_cmap(cmap_name, 15).colors[2:])
    layout = np.array([i for i in range(1,97)]).reshape(8,12).T
    layout = np.flipud(layout[:,8-ncols:])
    dflist = [np.log10(pd.read_csv(f"{root}{p}.csv",
                        index_col=0)/pd.read_csv(f"{root}{p}_fsc.csv",
                        index_col=0)) + 4 for p in details['plates']]
    df = pd.concat(dflist)

    paramsdflist = pd.concat([pd.read_csv(f'{root}extracted_params_{p}.csv',index_col='id') for p in details['plates']])
    paramsdf = paramsdflist.groupby(paramsdflist.index).mean()
    data = [d for d in df[str(1)].values if not np.isnan(d)]
    y, bins = np.histogram(data,bins=100)
    y = y/y.sum()
    xwin = (bins[1] - bins[0])/2
    x = [b + xwin for b in bins[:-1]]
    muThresh = paramsdf.muThresh.iloc[0]
    counter = 0
    extracted_params = np.array([estimated_on_fraction(x, row) for i, row in paramsdf.iterrows()])
    extracted_params = np.fliplr(np.flipud(extracted_params.reshape(8,12).T))[:,8-ncols:]
    for i in range(12):
        for j  in range(ncols):
            well = layout[i,j]
            wellc = extracted_params[i,j]
            data = [d for d in df[str(well)].values if not np.isnan(d)]
            y, bins = np.histogram(data,bins=100)
            y = y/y.sum()
            xwin = (bins[1] - bins[0])/2
            x = [b + xwin for b in bins[:-1]]
            axes[counter].plot(x,y,'k',lw=5)
            axes[counter].set_facecolor(cmap1(wellc))
            counter += 1

    for ax in axes:
        ax.set_xlim(0,5)
        ax.set_ylim(0,0.07)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()        
    plt.savefig(f'{details["outname"]}.pdf',dpi=300)        
        
    
