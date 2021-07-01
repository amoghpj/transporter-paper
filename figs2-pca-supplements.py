import pandas as pd
import matplotlib.pyplot as plt
stmap ={"a28y":"#d72631", "cd103":"#12a4d9", "sh07":"#5c3c92"}
plot_styles = {'fru':{'style':"^",'size':10},
               'mal':{'style':"v",'size':10},
               '02glc':{'style':"<",'size':10},
               '002glc':{'style':"P",'size':10},
               '2glc':{'style':"X",'size':8},
               'raf':{'style':"d",'size':10},
               'ac':{'style':"*",'size':10}}

paths = ["./data/allgenes-no-combine-expression-PCA.csv", "./data/allgenes-expression-PCA.csv"]

fig = plt.figure(figsize=(5,10))
axlist = [fig.add_subplot(2,1,i) for i in range(1,3)]
def plot_strains_nutrients(df, stmap, plot_styles, ax):
    df = df[~df.nutrient.str.contains("gal")]    
    normmap = {"2-glc":"2glc", "02-glc":"02glc","002-glc":"002glc",
               "2-fru":"fru","2-ace":"ac","2-mal":"mal","2-raf":"raf"}
    colors = [stmap[s] for s in df.strains]
    df["nutname"] = [normmap[n] for n in df.nutrient]
    for n,c, x, y in zip(df.nutname, colors, df.PC1, df.PC2):
        if n not in plot_styles.keys():
            print(n)
        ax.plot(x,y,marker=plot_styles[n]['style'],c=c,markersize=plot_styles[n]['size'])
        ax.annotate(n,xy=(x,y))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

for ax, p in zip(axlist, paths):
    df = pd.read_csv(p)
    plot_strains_nutrients(df, stmap, plot_styles, ax)
plt.tight_layout()
plt.savefig("s2-rnaseq-pca.pdf",dpi=300)
plt.close()
