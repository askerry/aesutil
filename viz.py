# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:45:21 2015

@author: amyskerry
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.graphics import regressionplots
import pandas as pd

def makelegend(legenddict, loc=(1,0)):
    patches=[]
    for key in legenddict.keys():
        patches.append(mpatches.Patch(color=legenddict[key], label=key))
    plt.legend(handles=patches, loc=loc)
    
def hierarchicalcluster(datamatrix, labels, similarity='euclidean', colorthresh='default'):
    '''plots dendrogram and returns clustering (item-1 x 4 array. first two columns are indices of clusters, 3rd column = distance between those clusters, 4th column = # of
  original observations in the cluster) and dend (dictionary of the data structures computed to render the
  dendrogram). see api here: http://hcluster.damianeads.com/cluster.html'''
    clustering=hcluster.linkage(datamatrix, metric=similarity)
    if colorthresh=='default':
        color_threshold=0.7*max(clustering[:,2]) #all descendents below a cluster node k will be assigned the same color if k is the first node below color_threshold. links connecting nodes with distances >= color_threshold are colored blue. default= 0.7*max(clustering[:,2])
    else: 
        color_threshold=colorthresh*max(clustering[:,2])
    fig = plt.figure()
    dend=hcluster.dendrogram(clustering, labels=labels, leaf_rotation=90, color_threshold=color_threshold)
    plt.tight_layout()
    return clustering, dend
    
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    if rgb[0]<=1:
        rgb=tuple([el*255 for el in rgb])
    return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
def partialplot(ycol, xcol, df, othercols=None):
    if othercols is None:
        othercols=[col for col in df.columns if col not in (ycol,xcol, 'Intercept')]
    f=regressionplots.plot_partregress(ycol, xcol, othercols, data=df, label_kwargs={'visible':False})
    
def screeplot():
   pass

def roccurve():
    pass

def residuals_vs_predicted(df, x, y):
    from pandas.stats.api import ols
    model = ols(x=df[x], y=df[y])
    df=pd.DataFrame(data={'residuals':model.resid, 'predicted {}'.format(y):model.y_fitted})
    sns.regplot('predicted {}'.format(y), 'residuals', df)