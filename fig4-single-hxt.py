from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.io import loadmat
import sys
import time
from tqdm import tqdm
from scipy.stats import gaussian_kde, norm
from sklearn import mixture

## 1. Read the metadata table
print("1. Reading metadata table...")
prefix = "../data/fig4-single-transporter/"
metadatatable = pd.read_csv(prefix + "fulldgmetatable.csv")
transporters = ["WT",
                "GAL2","HXT1", "HXT2", "HXT5", "HXT10"
                ]
platedata = metadatatable[metadatatable.useData == 1]
metadatatable["ind"] = list(range(metadatatable.shape[0]))
plateindex = {}

for t in transporters:
    plateindex[t] = metadatatable[metadatatable.plateNames == t].ind.values

## 2. Load the data file
print("2. Reading data table...")
starttime = time.time()
data = loadmat(prefix + "dgForPaper.mat",squeeze_me=False)
print(f"\tLoading took {time.time() - starttime}s")
plateNorm = data['normSSC']
cmap_name = "viridis"
## This sets estimated x and y offsets in the 16x14 grid for each plate type
positionmapper =  {
    "Standard double gradient":[5,0],# Correct
    "Low glucose full plate":[2,8],  # only two conditions have this plate
    "Low glucose half plate":[5,4],  # correct
    "Low glu-gal gradient":[0,1]      # adjust - correct
                   }
## Try to fill each well in a linear list
collect = {t:{ind:[]
              for ind in range(14*16)}
           for t in transporters}
layouts={
    "Standard double gradient":{1: 0, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11, 9:12, 10:13, 11:14, 12:15, 
                                13:16, 14:21, 15:22, 16:23, 17:24, 18:25, 19:26, 20:27, 21:28, 22:29, 23:30, 24:31, 
                                25:32, 26:37, 27:38, 28:39, 29:40, 30:41, 31:42, 32:43, 33:44, 34:45, 35:46, 36:47, 
                                37:48, 38:53, 39:54, 40:55, 41:56, 42:57, 43:58, 44:59, 45:60, 46:61, 47:62, 48:63, 
                                49:64, 50:69, 51:70, 52:71, 53:72, 54:73, 55:74, 56:75, 57:76, 58:77, 59:78, 60:79, 
                                61:80, 62:85, 63:86, 64:87, 65:88, 66:89, 67:90, 68:91, 69:92, 70:93, 71:94, 72:95, 
                                73:96, 74:101, 75:102, 76:103, 77:104, 78:105, 79:106, 80:107, 81:108, 82:109, 83:110, 84:111, 
                                85:208, 86:213, 87:214, 88:215, 89:216, 90:217, 91:218, 92:219, 93:220, 94:221, 95:222, 96:223},
    "Low glucose half plate":{1: 48 , 2: 53, 3:  54, 4:  55, 5:  56, 6:  57, 7:  58, 8:  59, 9:  60, 10: 61, 11: 62, 12: 63, 
                              13:64 , 14:69 , 15:70 , 16:71 , 17:72 , 18:73 , 19:74 , 20:75 , 21:76 , 22:77 , 23:78 , 24:79 , 
                              25:80 , 26:85 , 27:86 , 28:87 , 29:88 , 30:89 , 31:90 , 32:91 , 33:92 , 34:93 , 35:94 , 36:95 , 
                              37:96 , 38:101, 39:102, 40:103, 41:104, 42:105, 43:106, 44:107, 45:108, 46:109, 47:110, 48:111, 
                              49:112, 50:117, 51:118, 52:119, 53:120, 54:121, 55:122, 56:123, 57:124, 58:125, 59:126, 60:127, 
                              61:128, 62:133, 63:134, 64:135, 65:136, 66:137, 67:138, 68:139, 69:140, 70:141, 71:142, 72:143, 
                              73:144, 74:149, 75:150, 76:151, 77:152, 78:153, 79:154, 80:155, 81:156, 82:157, 83:158, 84:159, 
                              85:160, 86:165, 87:166, 88:167, 89:168, 90:169, 91:170, 92:171, 93:172, 94:173, 95:174, 96:175, },
    "Low glu-gal gradient":{1 :32,    2:33 ,     3:34,     4:35,     5:36,     6:37,     7:38,     8:39,
                             9:48,    10:49,11:50,  12:51,  13:52,  14:53,  15:54,    16:55,
                            17:64 , 18:65 , 19:66 , 20:67 , 21:68 , 22:69 , 23:70 , 24:71 , 
                            25:80 , 26:81 , 27:82 , 28:83 , 29:84 , 30:85 , 31:86 , 32:87 ,
                            33:96 , 34:97 , 35:98 , 36:99 , 37:100, 38:101, 39:102, 40:103,
                            41:112, 42:113, 43:114, 44:115, 45:116, 46:117, 47:118, 48:119, 
                            49:128, 50:129, 51:130, 52:131, 53:132, 54:133, 55:134, 56:135,
                            57:144, 58:145, 59:146, 60:147, 61:148, 62:149, 63:150, 64:151,
                            65:160, 66:161, 67:162, 68:163, 69:164, 70:165, 71:166, 72:167, 
                            73:176, 74:177, 75:178, 76:179, 77:180, 78:181, 79:182, 80:183,
                            81:192, 82:193, 83:194, 84:195, 85:196, 86:197, 87:198, 88:199,
                            89:208, 90:209, 91:210, 92:211, 93:212, 94:213, 95:214, 96:215, },
    "Low glucose full plate":{1: 48 , 2: 53, 3:  54, 4:  55, 5:  56, 6:  57, 7:  58, 8:  59, 9:  60, 10: 61, 11: 62, 12: 63, 
                              13:64 , 14:69 , 15:70 , 16:71 , 17:72 , 18:73 , 19:74 , 20:75 , 21:76 , 22:77 , 23:78 , 24:79 , 
                              25:80 , 26:85 , 27:86 , 28:87 , 29:88 , 30:89 , 31:90 , 32:91 , 33:92 , 34:93 , 35:94 , 36:95 , 
                              37:96 , 38:101, 39:102, 40:103, 41:104, 42:105, 43:106, 44:107, 45:108, 46:109, 47:110, 48:111, 
                              49:112, 50:117, 51:118, 52:119, 53:120, 54:121, 55:122, 56:123, 57:124, 58:125, 59:126, 60:127, 
                              61:128, 62:133, 63:134, 64:135, 65:136, 66:137, 67:138, 68:139, 69:140, 70:141, 71:142, 72:143, 
                              73:144, 74:149, 75:150, 76:151, 77:152, 78:153, 79:154, 80:155, 81:156, 82:157, 83:158, 84:159, 
                              85:160, 86:165, 87:166, 88:167, 89:168, 90:169, 91:170, 92:171, 93:172, 94:173, 95:174, 96:175, }# {1:,2:,3:,4:,5:,6:,7:,8:,9:,10:,11:,12:,
                              # 13:,14:,15:,16:,17:,18:,19:,20:,21:,22:,23:,24:,
                              # 25:,26:,27:,28:,29:,30:,31:,32:,33:,34:,35:,36:,
                              # 37:,38:,39:,40:,41:,42:,43:,44:,45:,46:,47:,48:,
                              # 49:,50:,51:,52:,53:,54:,55:,56:,57:,58:,59:,60:,
                              # 61:,62:,63:,64:,65:,66:,67:,68:,69:,70:,71:,72:,
                              # 73:,74:,75:,76:,77:,78:,79:,80:,81:,82:,83:,84:,
                              # 85,86:,87:,88:,89:,90:,91:,92:,93:,94:,95:,96:,}
}                                      

## 3. Collect the plates of interest
print("3. Collect wells...")
exceptionlist = []#("WT", "Low glucose half plate")]
for hxtid, hxt in enumerate(transporters):
    print(hxt)
    # filter to a given HXT
    for pt, (xoffset, yoffset) in positionmapper.items():
        print(pt)
        hxt_pt_plates = metadatatable[(metadatatable.plateNames == hxt) & (metadatatable.plateType == pt)]
        for i, row in hxt_pt_plates.iterrows():
            if pt == 'Standard double gradient':
                pdata = plateNorm[:, i][0]
                y_range = list(range(12))
                x_range = list(range(8))
                next_element = lambda x,y: x*12 + y + 1
            if pt == 'Low glucose half plate':
                pdata = np.fliplr(plateNorm[:, i][0])
                y_range = list(range(12))
                x_range = list(range(8))
                next_element = lambda x,y: x*12 + y + 1
            if pt == 'Low glucose full plate':
                pdata = np.fliplr(plateNorm[:, i][0])
                y_range = list(range(12))
                x_range = list(range(8))
                next_element = lambda x,y: x*12 + y + 1                
            if pt == "Low glu-gal gradient":
                pdata = np.fliplr(plateNorm[:, i][0].T)
                y_range = list(range(8))
                x_range = list(range(12))
                next_element = lambda x,y: x*8 + y + 1
            # cols = 12
            if (hxt, pt) not in exceptionlist:
                for x in x_range:
                    for y in y_range:
                        collect[hxt][layouts[pt][next_element(x,y)]].append(list(4 + np.log10(pdata[x,y]['yfp'])))

def gaussian_fit(x, y, vals, muThresh,ax=None):
    yhist, bins = np.histogram(vals,bins=100)
    xhist = [b + (bins[1] - bins[0])/2 for b in bins[:-1]]
    if ax is not None:
        ax.plot(xhist, yhist/sum(yhist),'k',lw=3)
    ## search for up to 2 components
    n_comp_max = 2
    models = []
    bic = []
    residuals = []
    # TODO Use residuals to do model selection in addition to BIC
    lowest = np.infty
    bestcomp = 1
    searchcomponents = [1 + i for i in range(n_comp_max)]
    for nc in searchcomponents:
        gmm = mixture.GaussianMixture(n_components = nc)
        gmm.fit(vals.reshape(-1, 1))
        gmmy = np.sum(
            np.array([a*norm.pdf(x,loc=m,scale=np.sqrt(s))
                      for a,m,s in zip(gmm.weights_,
                                       gmm.means_.flatten(),
                                       gmm.covariances_.flatten())]),
                      axis=0)        
        models.append(gmm)
        bic.append(gmm.bic(vals.reshape(-1, 1)))
        residuals.append(calculate_residual(y, gmmy))
        # if bic[-1] < lowest:
        #     lowest = bic[-1]
        #     bestgmm = gmm
        #     bestcomp = nc
    # model selection: now uses 
    bestmodelidx = residuals.index(min(residuals))
    bestgmm = models[bestmodelidx]
    mu_p = bestgmm.means_.flatten()
    sigma_p = bestgmm.covariances_.flatten()
    A_p = bestgmm.weights_
    ret = {}        
    if len(mu_p) == 2:
        if max(A_p) > 0.9:
            # unimodal
            ind = A_p.argmax()
            if mu_p[ind] > muThresh:
                ret["on_level"] = mu_p[ind]
                ret["off_level"] = 0.0 # np.nan
                ret["on_sigma"] = sigma_p[ind]
                ret["off_sigma"] = 0.0 # np.nan
                ret["on_fraction"] = 1.0
                ret["off_fraction"] = 0.0
            else:
                ret["on_level"] = 0.0
                ret["off_level"] = mu_p[ind]
                ret["on_sigma"] = 0.0
                ret["off_sigma"] = sigma_p[ind]
                ret["on_fraction"] = 0.0
                ret["off_fraction"] = 1.0
        else:
            #bimodal
            mu1, mu2 = mu_p
            if mu1 > mu2:
                rind = 0
                lind = 1
            else:
                rind = 1
                lind = 0
            # On fraction
            if mu_p[rind] >= muThresh and mu_p[lind] >= muThresh:
                ret["on_level"] = A_p[rind]*mu_p[rind] + A_p[lind]*mu_p[lind]
                ret["off_level"] = 0
                ret["on_sigma"] = sigma_p[rind] + sigma_p[lind]
                ret["off_sigma"] = 0
                # ret["on_fraction"] = A_p[rind]/(sum(A_p))
                # ret["off_fraction"] = A_p[lind]/(sum(A_p))
                ret["on_fraction"] = 1.0#A_p[rind]
                ret["off_fraction"] = 0.0#A_p[lind]
            elif mu_p[rind] >= muThresh and mu_p[lind] < muThresh:
                ret["on_level"] = mu_p[rind]
                ret["off_level"] = mu_p[lind]
                ret["on_sigma"] = sigma_p[rind]
                ret["off_sigma"] = sigma_p[lind]
                ret["on_fraction"] = A_p[rind]/(sum(A_p))
                ret["off_fraction"] = A_p[lind]/(sum(A_p))
                # ret["on_fraction"] = A_p[rind]
                # ret["off_fraction"] = A_p[lind]
            else:
                ret["on_level"] = 0.0
                ret["off_level"] = mu_p[lind]
                ret["on_sigma"] = 0.0
                ret["off_sigma"] = sigma_p[lind]
                ret["on_fraction"] = 0.0
                ret["off_fraction"] = 1.0
    elif len(mu_p) == 1:
        mu  = mu_p.flatten()[0]
        # On fraction
        if mu > muThresh:
            ret["on_level"] = mu_p[0]
            ret["on_sigma"] = sigma_p[0]            
            ret["on_fraction"] = 1.#A_p[0]            
            ret["off_fraction"] = 0.0
            ret["off_level"] = 0.0 # np.nan
            ret["off_sigma"] = 0.0 # np.nan        
        else:
            ret["on_fraction"] = 0.0
            ret["on_level"] = 0.0 # np.nan
            ret["on_sigma"] = 0.0 # np.nan
            ret["off_level"] = mu_p[0]
            ret["off_sigma"] = sigma_p[0]
            ret["off_fraction"] = 1.#A_p[0]
    # yfit = [[a*norm.pdf(_x, loc=m, scale=np.sqrt(s))\
    #          for m,s,a in zip(mu_p, sigma_p, A_p)] for _x in x]
    actual_on_fraction = 0
    if ret['off_sigma'] > 0:
        yfit_off = ret['off_fraction']*norm.pdf(x, loc=ret['off_level'],
                                        scale=np.sqrt(ret['off_sigma']))
        denom = sum(yfit_off)
        num = sum([ret['off_fraction']*norm.pdf(_x, loc=ret['off_level'],
                                              scale=np.sqrt(ret['off_sigma']))
                   for _x in x if _x > muThresh])            
        actual_on_fraction += (num/denom)*ret['off_fraction']
        if ax is not None:
            #ax.plot(x, yfit_off/sum(yfit_off), 'r--')
            ax.plot(x, gmmy/sum(gmmy), 'r--')            
    if ret['on_sigma'] > 0:
        yfit_on = ret['on_fraction']*norm.pdf(x, loc=ret['on_level'],
                                        scale=np.sqrt(ret['on_sigma']))            
        # Now, the actual fraction of ON cells has to take the spread into account
        denom = sum(yfit_on)
        num = sum([ret['on_fraction']*norm.pdf(_x, loc=ret['on_level'],
                                              scale=np.sqrt(ret['on_sigma']))
                   for _x in x if _x > muThresh])
        actual_on_fraction += num/denom*ret['on_fraction']
        if ax is not None:
            ax.axvline(ret['on_level'])            
            ax.plot(x, gmmy/sum(gmmy), 'k--')                    
    return(ret, actual_on_fraction)

def calculate_residual(y1,y2):
    return(np.mean(np.power(np.array(y1),2) -  np.power(np.array(y2),2)))

## 4. Plot!
def make_on_fraction_comparison(collect, transporters):
    print("4. Plotting on fraction comparison")
    cmap1 = matplotlib.colors.ListedColormap(cm.get_cmap(cmap_name, 30).colors)
    fig = plt.figure(figsize=(6,4))
    axes = [fig.add_subplot(2,3,i) for i in range(1,7)]
    for hxtid, hxt in enumerate(transporters):
        wellmeans = np.zeros((14*16))
        print("Caculating population histograms")
        mean = 0
        x = np.linspace(0, 5., 100)        
        for i in tqdm(range(14*16)):
            L = []
            for l in collect[hxt][i]:
                if len(l) > 0:
                    # NOTE There is some funny business with how the matlab data file
                    #      stores the values. They are stored as a python list of single value
                    #      numpy arrays. This seems to be tripping up the gaussian_kde function.
                    #      I get around this by converting the list of values to an array and then
                    #      flattening it.
                    L.extend(np.array(l).flatten('C'))
            if len(L)> 0:
                L = np.array(L)
                kernel = gaussian_kde(L, bw_method = 0.3) 
                y = kernel(x)
                onThreshold = 2.0
                ext, aof = gaussian_fit(x, y, L, onThreshold)
                wellmeans[i] = aof#ext['on_fraction']
            else:
                wellmeans[i] = wellmeans[max(0, i-16)]
            
        print("Plotting histograms")
        minpopmean = wellmeans[16*13]
        maxpopmean = wellmeans[16*14 -1]
        wellmeanstable = np.reshape(wellmeans,(14,16))
        # wellmeanstable = (maxpopmean - wellmeanstable)/((maxpopmean - minpopmean))
        axes[hxtid].imshow(wellmeanstable, cmap = cmap1)
        axes[hxtid].annotate(hxt,xy=(0,2),fontsize=16,color="w")
        axes[hxtid].set_xticks([])
        axes[hxtid].set_yticks([])            
    plt.tight_layout()    
    plt.savefig(f"./img/full_dg_comparison.png",dpi=300)
    
def make_on_fraction_comparison_display(collect, transporters):
    print("4. Plotting on fraction comparison")
    cmap1 = matplotlib.colors.ListedColormap(cm.get_cmap(cmap_name, 30).colors)
    for hxtid, hxt in enumerate(transporters):
        fig = plt.figure(figsize=(32,28))
        axes = [fig.add_subplot(14,16,i) for i in range(1,14*16+1)]        
        wellmeans = np.zeros((14*16))
        print("Caculating population histograms")
        mean = 0
        x = np.linspace(0, 5., 100)        
        for i in tqdm(range(14*16)):
            L = []
            for l in collect[hxt][i]:
                if len(l) > 0:
                    # NOTE There is some funny business with how the matlab data file
                    #      stores the values. They are stored as a python list of single value
                    #      numpy arrays. This seems to be tripping up the gaussian_kde function.
                    #      I get around this by converting the list of values to an array and then
                    #      flattening it.
                    L.extend(np.array(l).flatten('C'))
            if len(L)> 0:
                L = np.array(L)
                kernel = gaussian_kde(L, bw_method = 0.3) 
                y = kernel(x)
                onThreshold = 2.0
                ext, aof = gaussian_fit(x, y, L, onThreshold)
                wellmeans[i] = aof#ext['on_fraction']
            else:
                wellmeans[i] = wellmeans[max(0, i-16)]
            axes[i].set_facecolor(cmap1(wellmeans[i]))
            
        # minpopmean = wellmeans[16*13]
        # maxpopmean = wellmeans[16*14 -1]
        # wellmeanstable = np.reshape(wellmeans,(14,16))
        # wellmeanstable = (maxpopmean - wellmeanstable)/((maxpopmean - minpopmean))
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0,5)
        plt.tight_layout()    
        plt.savefig(f"./img/{hxt}_dg_comparison.png",dpi=300)
        plt.close()
    
def make_decision_threshold_comparison():    
    cmap1 = matplotlib.colors.ListedColormap(cm.get_cmap(cmap_name, 30).colors)
    fig = plt.figure(figsize=(6,4))
    axes = [fig.add_subplot(2,3,i) for i in range(1,7)]
    for hxtid, hxt in enumerate(transporters):
        print(hxtid)
        #fig = plt.figure(figsize=(32, 28))
        #axes = [fig.add_subplot(14,16,ind) for ind in range(1,14*16+1)]
        print(hxt)
        wellmeans = np.zeros((14*16))
        print("Caculating population histograms")
        mean = 0
        for i in tqdm(range(14*16)):
            L = []
            for l in collect[hxt][i]:
                if len(l) > 0:
                    vals, bins = np.histogram( l,bins=100)
                    ynorm = vals/sum(vals)
                    w = (bins[1] - bins[0])/2.
                    xpos = [b + w for b in bins[:-1]]
                    #axes[i].plot(xpos,ynorm,'white')
                    L.extend(l)
            if len(L)> 0:
                mean = np.mean(L)
            else:
                mean = wellmeans[max(0, i-16)]
            wellmeans[i] = mean
        print("Plotting histograms")
        minpopmean = wellmeans[16*13]
        maxpopmean = wellmeans[16*14 -1]
        wellmeanstable = np.reshape(wellmeans,(14,16))
        wellmeanstable = (maxpopmean - wellmeanstable)/((maxpopmean - minpopmean))
        # ax = fig.add_subplot(1,1,1)
        axes[hxtid].imshow(wellmeanstable, cmap = cmap1)
        axes[hxtid].annotate(hxt,xy=(0,2),fontsize=16,color="w")
        axes[hxtid].set_xticks([])
        axes[hxtid].set_yticks([])            
        # for i in tqdm(range(14*16)):
        #     axes[i].set_facecolor(cmap1((maxpopmean-wellmeans[i])/(maxpopmean - minpopmean)))         
        # for ax in axes:
        #     ax.set_xlim(0,5.0)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.tight_layout()
        # plt.savefig(f"./img/2021-06-08-{hxt}_full_dg.png",dpi=100)
        # plt.close()
    plt.tight_layout()    
    plt.savefig(f"./img/full_dg_comparison.png",dpi=300)    

make_on_fraction_comparison(collect, transporters)
#make_on_fraction_comparison_display(collect, transporters)    
