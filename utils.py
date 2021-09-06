import pandas as pd 
import sys
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wrenlab.matrix import mmat
from wrenlab.normalize import quantile

sys.path.insert(0,"/home/xiavan/gitlab/mana/")
import mana
import utils 

#hardcoded data fetching helper
def load_mana():
    path="/data/ncbi.bak/geo/GPL13534/meta/meth_tools/mana/mana_GEO.full_labels"
    samples = pd.read_csv(path,sep=",",header=0,index_col=0)
    print(samples.shape)
    path = "/data/ncbi.bak/geo/GPL13534/GPL13534.matrix.mmat"
    XD = mmat.MMAT(path)
    X = XD.loc[samples.index,:].to_frame()
    print(X.shape)
    X = X.fillna(X.mean())
    X = quantile(X.T).T
    return(X)

def load_gb():
    path = "/data/backup/xiavan/mana_gb_means.h5"
    data = pd.read_hdf(path,key="data")
    return(data.T)

def load_tss():
    path = "/data/backup/xiavan/mana_tss1500_means.h5"
    data = pd.read_hdf(path,key="data")
    return(data.T)

def load_go():
    path = "./go.h5"
    go = pd.read_hdf(path,key="data")
    #path="./go_genes.csv"
    #go = pd.read_csv(path,sep=",",header=0,index_col=0)
    return(go.T)


def sample():
    data = load_mana()
    variance = data.var()
    top = variance.sort_values(ascending=False).head(int(variance.shape[0]/10))
    data = data[top]
    
    meta = "/data/ncbi.bak/geo/GPL13534/meta/meth_tools/mana/mana_GEO.full_labels"
    meta = pd.read_csv(meta,sep=",",header=0,index_col=0)
    tissues = meta.TissueName.value_counts()[2:7]
    tissues = meta[meta.TissueName.isin(tissues.index)]
    data = data.loc[tissues.index]
    print("final dims:",data.shape)
    return(data)

def fetch_data():
    data = "/data/backup/xiavan/mana_dl_sample.h5"
    data = pd.read_hdf(data,key="data")
    return(data)

def fetch_meta():
    meta = "/data/ncbi.bak/geo/GPL13534/meta/meth_tools/mana/mana_GEO.full_labels"
    meta = pd.read_csv(meta,sep=",",header=0,index_col=0)
    return(meta)
  
