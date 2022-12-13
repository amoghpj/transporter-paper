import numpy as np
from scipy.stats import norm
from matplotlib import cm
import matplotlib

cmap_name = "viridis"
cmap1 = matplotlib.colors.ListedColormap(cm.get_cmap(cmap_name, 30).colors[5:])

def estimated_on_fraction(params, x, muThresh):
    estimated_of = 0.0
    on_level = params['on_level']
    off_level = params['off_level']
    on_fraction = params['on_fraction']
    off_fraction = params['off_fraction']
    on_sigma = params['on_sigma']
    off_sigma = params['off_sigma']
    estimated_of = 0
    ## For the estimated OFF population characteristics, get statistics on population exceeding the cutoff
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
