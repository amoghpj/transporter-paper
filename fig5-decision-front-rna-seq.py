import numpy as np
import pprint
import sys
import pandas as pd
import glob as glob
import seaborn as sns
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse
from sklearn.linear_model import LinearRegression
from itertools import product
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
import matplotlib
font = {'size'   : 15}
matplotlib.rc('font', **font)
### Plot setup
fig = plt.figure(figsize=(15,20))
gs = fig.add_gridspec(nrows=5,ncols=3)
axlist = [fig.add_subplot(gs[i,:1]) for i in range(0,3)]
axpheno = fig.add_subplot(gs[4,:1])
axrna = fig.add_subplot(gs[3,1:])
axcomp = fig.add_subplot(gs[4,1:])

#### Plot decision fronts
sd = defaultdict(lambda: defaultdict(list))
coords = namedtuple("Coords", ["x","y"])

def find_first_bimodal_well(data, nut):
    collect = []
    for row in range(8):
        for well in [row*12 + i for i in range(12)]:
            if well in df.index and data.iloc[well].on_fraction >= 0.5:
                    collect.append(well)
                    break
    return(collect)

def get_coords(l, data):
    return([coords(data.iloc[l_].gal ,data.iloc[l_].glc) for l_ in l])
normmap = {"2-glc":"2glc", "02-glc":"02glc","002-glc":"002glc",
           "2-fru":"fru","2-ace":"ac","2-mal":"mal","2-raf":"raf"}

plot_styles = {'fru':{'style':"^",'size':10,"label":"2FRU"},
               'mal':{'style':"v",'size':10,"label":"2MAL"},
               '02glc':{'style':"<",'size':10,"label":"0.2GLC"},
               '002glc':{'style':"P",'size':10,"label":"0.02GLC"},
               '2glc':{'style':"X",'size':8,"label":"2GLC"},
               'raf':{'style':"d",'size':10,"label":"2RAF"},
               'ac':{'style':"*",'size':10,"label":"2AC"}}
straindict = sd
titlemap = {"a28y":"WT (S288c)", "sh07":"Sensorless (CEN.PK)", "cd103":"WT (CEN.PK)"}

files = glob.glob("./data/sensorless-wildtype-decision-front/*.csv")

for f in files:
    pathcomp = f.split("/")[3].split("-")
    strain = pathcomp[0].split("_")[2]
    straindict[strain][pathcomp[1]].append(f)

strains = ["sh07","cd103","a28y"]
nutconditions = ["2glc","fru", "02glc", "002glc", "mal","raf","ac"]
for i,strain in enumerate(strains):
    ax = axlist[i]
    data = straindict[strain]
    for nut in nutconditions:
        dataset = data[nut]
        allcoords = []
        extent = 2
        for dat in dataset[:extent]:
            df = pd.read_csv(dat)
            firstwell = find_first_bimodal_well(df, nut)
            allcoords.extend(get_coords(firstwell, df))
        ycoords = sorted(list(set([c.y for c in allcoords])))
        glcvals = {y:[] for y in ycoords}
        for c in allcoords:
            glcvals[c.y].append(c.x)
        for k, v in glcvals.items():
            if len(v) == 1:
                glcvals[k].append(v[0])

        X = [np.mean(np.log2(np.array(glcvals[y]))) for y in glcvals.keys()]
        Y = [np.log2(y) for y in glcvals.keys()]
        reg = LinearRegression().fit(np.array(X[1:]).reshape(-1,1), Y[1:])
        err = []
        for y, xlist in glcvals.items():
            mean = np.mean(np.log2(np.array(xlist)))
            alldiff = mean - np.log2(np.array(xlist))
            min_ = abs(min(alldiff))
            max_ = abs(max(alldiff))
            err.append([min_, max_])
        err= np.array(err).T
        s=reg.score(np.array(X[1:]).reshape(-1,1), Y[1:])
        label = f"%s (%d)" % (nut, len(dataset))
        m = 'o'
        ms = 5
        if nut in plot_styles.keys():
            m = plot_styles[nut]['style']
            ms = plot_styles[nut]['size']
        ax.errorbar(X[1:], Y[1:], xerr=err[:,1:],
                    marker=m,ms=ms, alpha=.9,
                    label=label) 
for i, (ax,strain) in enumerate(zip(axlist,strains)):
    if i == 2:
        ax.set_xlabel("log$_2$(% Gal)")
        leg = ax.legend()# bbox_to_anchor=(0.5,-0.1)
        leg.get_frame().set_alpha(0)
    ax.set_ylabel("log$_2$(% Glc)")    
    ax.set_yticks([0,-2,-4,-6,-8])
    ax.set_yticklabels(["0","-2","-4","-6",""])
    ax.set_ylim(-7.5,0.5)    
    ax.set_xlim(-10.5,-0.5)    
    ax.annotate(titlemap[strain], xy=(-8,0))# TODO ,fontsize=16)
    ax.grid()

#### Process phenotype data
conditionids = [f"{n}-{s}".replace("%","").replace(".","") 
                for s, n in product(strains, nutconditions)]
collect = {c:[] for c in conditionids}
metric = "on_fraction"
for (s, n), cid in zip(product(strains, nutconditions),conditionids):
    replicates = []
    dataset = straindict[s]
    for _f in dataset[n]:
        df = pd.read_csv(_f)
        df = df.replace(np.nan, 0.0)
        vals = [0 for _ in range(96)]
        for i, row in df.iterrows():
           vals[int(row["id"])-1] = row[metric]
        replicates.append(vals)
    if len(replicates) == 3:
        for v1, v2, v3 in zip(replicates[0], replicates[1], replicates[2]):
            collect[cid].append(np.mean([v1,v2,v3])) ############### IMPORTANT
    else:
        for v1, v2 in zip(replicates[0], replicates[1]):
            collect[cid].append(np.mean([v1,v2])) ############### IMPORTANT
collectdf = pd.DataFrame(collect)
collectdf.rename({f"{n}-a28":f"{n}-a28y" for n in nutconditions},axis="columns",inplace=True)
collectdf.to_csv("./data/phenotypic_data.csv")

pc = PCA(n_components=2)
PC = pc.fit_transform(collectdf.values.T)

axpheno.set_xlabel("PC2 - Decision Front")
axpheno.set_ylabel("PC1 - Decision Front")
stmap ={"a28y":"#d72631", "cd103":"#12a4d9", "sh07":"#5c3c92"}
colors = [stmap[c.split("-")[1]] for c in collectdf.columns]
nut = [c.split("-")[0] for c in collectdf.columns]
for n,c, x, y in zip(nut,colors,PC[:,1],PC[:,0]):
    axpheno.plot(x,y,marker=plot_styles[n]['style'],markersize=plot_styles[n]['size'],c=c)
axpheno.set_ylabel("PC1 - Decision Front")#
axpheno.set_xlim(-2,2.2)
axpheno.set_ylim(-3,3.5)
axpheno.set_title("GAL Decision Front")

#### Transcriptional barplots
countdf = pd.read_csv("./data/raw-counts/raw-counts-all.csv",index_col=0)
countdf = countdf.loc[[ind for ind in list(countdf.index) if "HXT" in ind]]
unique = list(set(["-".join(c.split("-")[:-1]) for c in countdf.columns]))
expdf_col = pd.DataFrame(columns=unique,index=countdf.index)
for u in unique:
    coloi = [c for c in countdf.columns if c.startswith(u)]
    expdf_col[u] = countdf[coloi].mean(axis='columns')
expdf_col = expdf_col.replace(np.nan, 0.0)
expdf = np.log10(expdf_col +1)
expdf = expdf[[c for c in expdf.columns if "gal" not in c]]
expdfstrains = [c.split("-")[-1] for c in expdf.columns]

expdfnutrient = [normmap["-".join(c.split("-")[:-1])] for c in expdf.columns]
expdfnormalized = [n.split("-")[2] + "-" + normmap["-".join(n.split("-")[:2])] for n in expdf.columns]
rnaseq_pca = pc.fit_transform(expdf.T.values)
sload, shxt = zip(*sorted(zip(pc.components_[0], expdf.index)))
shxt = list(reversed(shxt))

# Now compute mean difference between rich and poor carbon sources...
expdf['meandiff'] = expdf[['2-glc-cd103','2-fru-cd103','02-glc-cd103']].median(axis=1) -\
    expdf[["002-glc-cd103", "2-mal-cd103","2-raf-cd103","2-ace-cd103"]].median(axis=1)
# ... sort those values and store the row order, which will be used to order the barplot below
expdforder = list(expdf.sort_values(by='meandiff', axis=0,ascending = False).index)
expdf = pd.melt(expdf.reset_index(), id_vars=['index'])

expdf['nut'] = ["-".join(v.split('-')[:2]) for v in expdf.variable]
axlist = [fig.add_subplot(gs[i,1:]) for i in range(3)]
order = ["2-glc", "2-fru","02-glc", "002-glc", "2-mal","2-raf","2-ace"]
for i, (s, ax) in enumerate(zip(strains, axlist)):
    coidf = expdf[expdf.variable.str.contains(s)]
    sns.barplot(x='index',y='value',hue='nut',
                order=shxt,#expdforder,
                hue_order=order,
                data=coidf,ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.set_ylim(1,5.5)
    ax.set_ylabel(s)
    ax.set_xlabel('')
    if i != 2:
        le = ax.legend()
        le.remove()
        ax.set_xticks([])
    else:
        leg = ax.legend()
        leg.get_frame().set_alpha(0)

#### Transcriptional PCA
rnaseqdf = pd.DataFrame({"PC1":rnaseq_pca[:,0],
                         "PC2":rnaseq_pca[:,1],
                         "strains":expdfstrains,
                         "nutrient":expdfnutrient,
                         "normalized":expdfnormalized})
colors = [stmap[s] for s in rnaseqdf.strains]
for n,c, x, y in zip(rnaseqdf.nutrient, colors, rnaseqdf.PC1, rnaseqdf.PC2):
    axrna.plot(x,y,marker=plot_styles[n]['style'],c=c,markersize=plot_styles[n]['size'])
xdelta = -0.0
ydelta = -0.0
for c, x, y in zip(rnaseqdf.nutrient,rnaseqdf.PC1,rnaseqdf.PC2):
    axrna.annotate(plot_styles[c]["label"],
                   xy=(x+xdelta,y+ydelta),
                   fontsize=14,
                   rotation=20,alpha=0.5)
axrna.set_title("HXT expression")
axrna.set_ylabel("PC2 - HXT expression")
axrna.set_xlabel("PC1 - HXT expression")
axrna.set_xlim(-2.2,3.7)

#### PC1 comparison
ordered1 = []
ordered2 = []
axcomp.set_yticks([])
axcomp.set_xlabel('PC1 - HXT expression')
axcompt = axcomp.twinx()
axcompt.set_ylim(-3,3.5)
axcompt.set_xlim(-2.2,3.7)
axcompt.set_ylabel('PC1 - Decision Front')
for s, n in product(strains, nutconditions):
    ordered1.append(rnaseqdf[rnaseqdf.normalized == f"{s}-{n}"].PC1.values[0])
    ordered2.append(rnaseqdf[rnaseqdf.normalized == f"{s}-{n}"].PC2.values[0])
colors = [stmap[c.split("-")[1]] for c in collectdf.columns]
for n,c, x, y in zip(nut, colors, ordered1, PC[:,0]):
    axcompt.plot(x,y,marker=plot_styles[n]['style'],c=c,markersize=plot_styles[n]['size'])
pccompdf = pd.DataFrame({'pcpheno':PC[:,0],
                         'pcrna':ordered1,
                         'strains':[c.split("-")[1] for c in collectdf.columns],
                         'nutrient':[c.split("-")[0] for c in collectdf.columns]})

annotcoords={'sh07':(-1,2),'cd103':(-0.7,-1.5),'a28y':(0.5,-0.8)}
for s in strains:
    dfoi = pccompdf[pccompdf.strains == s]
    r, p = pearsonr(dfoi.pcpheno, dfoi.pcrna)
    axcompt.annotate(f"r={round(r,2)}",xy=annotcoords[s],color=stmap[s],fontsize=24)
    dfoi.to_csv(f"./data/{s}-pheno-rna.csv")

e1 = Ellipse(xy=(0.5,-1.8), width=1.5, height=6.5, angle = 60, color=stmap["cd103"],alpha=0.1)
e2 = Ellipse(xy=(0.5,-0.5), width=1, height=7, angle = 60, color=stmap["a28y"],alpha=0.1)
e3 = Ellipse(xy=(-0.25,2.2), width=4, height=2., color=stmap["sh07"],alpha=0.1)
for e in [e1,e2,e3]:
    axcompt.add_artist(e)
ind = 19
x1, y1, x2 = PC[ind,1],PC[ind,0],ordered1[ind]
xyA = (x2,y1)
xyB = (x1,y1)
con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                      axesA=axcompt, axesB=axpheno, color="k",alpha=0.2)
con.set_in_layout(False)
axcomp.add_artist(con)
x1, y1,x2 = ordered1[ind], ordered2[ind], PC[ind,0]
xyB = (x1,y1)
xyA = (x1,x2)
con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                      axesA=axcompt, axesB=axrna, color="k",alpha=0.2)
con.set_in_layout(False)
axcomp.add_artist(con)

plt.tight_layout()
plt.savefig("decision-threshold-2.pdf",dpi=400)
plt.close()
