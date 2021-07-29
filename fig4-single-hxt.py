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
h = os.path.expanduser("~")
## 1. Read the metadata table
print("1. Reading metadata table...")
prefix = "../data/fig4-single-transporter/"
metadatatable = pd.read_csv(prefix + "fulldgmetatable.csv")
transporters = ["WT",  "GAL2","HXT1", "HXT2", "HXT5", "HXT10",]
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
cmap_name = "viridis_r"
## This sets estimated x and y offsets in the 16x14 grid for each plate type
positionmapper =  {
    "Standard double gradient":[5,0],# Correct
    # "Low glucose full plate":[2,8],  # only two conditions have this plate
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
                            89:208, 90:209, 91:210, 92:211, 93:212, 94:213, 95:214, 96:215, }    
}                                      

def normaloffset(xoffset, yoffset, counter,cols):
    ind = xoffset + 2*(counter//12) + yoffset*14 + counter
    return(ind)
def glcgaloffset(xoffset, yoffset, counter,cols):
    return(xoffset + (14-cols)*(counter//cols) + yoffset*14 + counter)

## 3. Collect the plates of interest
print("3. Collect wells...")

for hxt in transporters:
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
            if pt == "Low glu-gal gradient":
                pdata = np.fliplr(plateNorm[:, i][0].T)
                y_range = list(range(8))
                x_range = list(range(12))
                next_element = lambda x,y: x*8 + y + 1
            # cols = 12
            for x in x_range:
                for y in y_range:
                    collect[hxt][layouts[pt][next_element(x,y)]].append(list(4 + np.log10(pdata[x,y]['yfp'])))
## 4. Plot!
print("4. Plotting")
cmap1 = matplotlib.colors.ListedColormap(cm.get_cmap(cmap_name, 20).colors[5:])
mean = 0
for hxt in transporters:
    fig = plt.figure(figsize=(32, 28))
    axes = [fig.add_subplot(14,16,ind) for ind in range(1,14*16+1)]
    print(hxt)
    for i in tqdm(range(14*16)):
        L = []
        for l in collect[hxt][i]:
            if len(l) > 0:
                vals, bins = np.histogram( l,bins=100)
                ynorm = vals/sum(vals)
                w = (bins[1] - bins[0])/2.
                xpos = [b + w for b in bins[:-1]]
                axes[i].plot(xpos,ynorm,'white')
                L.extend(l)
        if len(L)> 0:
            mean = np.mean(L)
        axes[i].set_facecolor(cmap1((4.-mean)/(4.)))            
    for ax in axes:
        ax.set_xlim(0,5.0)
        #ax.set_ylim(0,1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"./img/2021-06-08-{hxt}_full_dg.png")
    plt.close()            
