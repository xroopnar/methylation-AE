import pandas as pd
import sys
#setting pyensembl cache dir 
import os 
import pyensembl as pe
import pybedtools

os.environ['PYENSEMBL_CACHE_DIR'] = '/data/backup/xiavan/pyensembl/cache'
from wrenlab import ensembl as we
from wrenlab.ncbi import gene
from wrenlab import enrichment
from wrenlab.matrix.mmat import MMAT
from wrenlab.normalize import quantile

from sklearn.linear_model import LogisticRegression as LogR
from sklearn.decomposition import PCA 
import metalearn

import sklearn
import wrenlab
import io

#sys.path.insert(0,"/home/xiavan/gitlab/mana/")
#import mana 
#import utils

def load_methylation(n):
    path = "/data/ncbi.bak/geo/GPL13534/GPL13534.matrix.mmat"
    XD = MMAT(path)
    X = XD.loc[XD.index[:n],:].to_frame()
    X = X.fillna(X.mean())
    X = wrenlab.normalize.quantile(X.T).T
    return(X)

def load_mana():
    samples = "/data/ncbi.bak/geo/GPL13534/meta/meth_tools/mana/mana_GEO.full_labels"
    samples = pd.read_csv(samples,sep=",",header=0,index_col=0)
    print(samples.shape)
    samples = list(samples.index)
    path = "/data/ncbi.bak/geo/GPL13534/GPL13534.matrix.mmat"
    XD = MMAT(path)
    X = XD.loc[samples,:].to_frame()
    print(X.shape)
    X = X.fillna(X.mean())
    X = wrenlab.normalize.quantile(X.T).T
    return(X)

def load_debug(n=1000):
    data = load_methylation(n)
    return(data)

def load_expression(n):
    path = "/data/ncbi.bak/geo/GPL570.mmat"
    XD = MMAT(path)
    X = XD.loc[XD.index[:n],:].to_frame()
    X = X.fillna(X.mean())
    X = wrenlab.normalize.quantile(X.T).T
    return(X)

def get_gdict():
    gmap = utils.fetch_570_map()
    gmap = gmap.drop_duplicates(subset="probe")
    gdict = pd.DataFrame(gmap.entrez)
    gdict.index = gmap.probe
    gdict = gdict.to_dict()
    gdict = gdict["entrez"]
    return(gdict)
        #data = data.groupby(gdict,axis=1).mean()
        
def load_expression_debug(n=1000,collapse=True):
    data = load_expression(n)
    print("raw dims:",data.shape)
    if collapse==True:
        gmap = utils.fetch_570_map()
        gmap = gmap.drop_duplicates(subset="probe")
        gdict = pd.DataFrame(gmap.entrez)
        gdict.index = gmap.probe
        gdict = gdict.to_dict()
        gdict = gdict["entrez"]
        data = data.groupby(gdict,axis=1).mean()
        data.columns = data.columns.astype(int)
    return(data)

def collapse_genes(x,method="mean"):
    if method == "mean":
        out = x.mean()
    if method == "median":
        out = x.median()
    #binarize the input probes and take the median value
    if method == "chalm":
        out = x.round(0).median()
    if method == "chalm2":
        out = (x>.5).astype(int).mean()
    return(out)

def to_bedtool(df):
    # assume df = name, chrom, start, end
    df = df.copy()
    df.columns = ["name", "chrom", "start", "end", "strand"]
    df = df.dropna()
    df["score"] = ""
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df = df.loc[:,["chrom", "start", "end", "name", "score", "strand"]]
    with io.StringIO() as h:
        df.to_csv(h, sep="\t", index=None, header=False)
        return pybedtools.BedTool(h.getvalue(), from_string=True)

#get back aligned annotations 
def map_annotations(X):
    A = enrichment.annotation("GO", taxon_id=9606).prune(min_element_count=25, max_element_count=500).matrix()
    X,A = X.align(A,join="inner",axis=1)
    return(X,A)

def to_pc(X,n=20):
    pca = PCA(n_components=n)
    pc = pca.fit_transform(X)
    pc = pd.DataFrame(pc,index=X.index)
    return(pc)

#first do go map on axis=1 of samples x probes
#then X=X.T

#remember this needs to be probes x samples
#will collapse to genes x samples 
#then u can directly pca to genes x PCs
class ensembl_loci(object):
    def __init__(self,data=None):
        self.data = data
        probes = "/data/ncbi.bak/geo/GPL13534/meta/GPL13534-11288.txt"
        probes = pd.read_csv(probes,skiprows=37,sep="\t",index_col=0)
        probes = probes[["Name","Chromosome_36","RANGE_START","RANGE_END"]]
        probes.columns = ["ProbeID","chrom","start","end"]
        probes["strand"] = ""
        probes = to_bedtool(probes)
        self.probes = probes
        loci = wrenlab.ensembl.loci(9606,version=75)
        #save df version to change later if you need it 
        self.loci = loci 
        self.genes = to_bedtool(loci.reset_index())
        df = gene.ensembl(9606)
        self.ensembl_entrez = dict(zip(df["Ensembl Gene ID"], df["GeneID"]))
        
    #window relative to the start (l) to end of gene (r)
    def get_gene_window(self,l,r):
        intersection = self.genes.window(self.probes, l=l, r=r, sw=True)
        locus_gene = {obj.fields[8]:self.ensembl_entrez[obj.fields[3]] for obj in intersection if obj.fields[3] in self.ensembl_entrez }
        return(locus_gene)
    
    def gene_window_collapse(self,l,r,method="mean"):
        locus_gene = self.get_gene_window(l=l,r=r)
        out = self.data.groupby(locus_gene).apply(lambda x : collapse_genes(x,method=method))
        return(out)
    
    def get_tss_window(self,l,r):
        genes = self.loci
        genes["end"] = genes["start"]
        genes = to_bedtool(genes.reset_index())
        intersection = genes.window(self.probes, l=l, r=r, sw=True)
        locus_gene = {obj.fields[8]:self.ensembl_entrez[obj.fields[3]] for obj in intersection if obj.fields[3] in self.ensembl_entrez }
        #locus_gene = self.get_gene_window(l,r,genes)
        return(locus_gene)
    
    def tss_window_collapse(self,l,r,method="mean"):
        locus_gene = self.get_tss_window(l,r)
        out = self.data.groupby(locus_gene).apply(lambda x : collapse_genes(x,method=method))
        return(out)

    def tss20002000(self,method="mean"):
        out = self.tss_window_collapse(l=2000,r=2000,method=method)
        return(out)
    
    def tss1500(self,method="mean"):
        out = self.tss_window_collapse(l=1500,r=0,method=method)
        return(out)
    
    def tss200(self,method="mean"):
        out = self.tss_window_collapse(l=200,r=0,method=method)
        return(out)
    
    def tss5050(self,method="mean"):
        out = self.tss_window_collapse(l=50,r=50,method=method)
        return(out)
    
    def gb(self,method="mean"):
        out = self.gene_window_collapse(l=0,r=0,method=method)
        return(out)
    
    def extended_gb(self,method="mean"):
        out = self.gene_window_collapse(l=1500,r=500,method=method)
        return(out)
    
    def exon():
        pass
    
    def intron():
        pass    

class function_prediction(object):
    def __init__(self,X):
        self.X = X
        #self.X,self.y = map_annotations(X)
        #print("mapped X dims:",X.shape)
        #print("mapped y dims:",y.shape)

    def tss20002000_qc(self,method):
        e = ensembl_loci(self.X)
        tss = e.tss20002000(method=method)
        print("gb mapped dim:",tss.shape)
        self.Xc = to_pc(tss)
        print("pc dim:",self.Xc.shape)
        self.Xc,self.y = map_annotations(self.Xc.T)
        print("go mapped X dims:",self.Xc.shape)
        print("go mapped y dims:",self.y.shape)
        qc_y = self.y.sample(1000)
        #return(self.Xc,qc_y)
        results = self.evaluate(self.Xc.T,qc_y.T)
        return(results)
        
    def tss1500_qc(self,method):
        e = ensembl_loci(self.X)
        tss = e.tss1500(method=method)
        print("gb mapped dim:",tss.shape)
        self.Xc = to_pc(tss)
        print("pc dim:",self.Xc.shape)
        self.Xc,self.y = map_annotations(self.Xc.T)
        print("go mapped X dims:",self.Xc.shape)
        print("go mapped y dims:",self.y.shape)
        qc_y = self.y.sample(1000)
        #return(self.Xc,qc_y)
        results = self.evaluate(self.Xc.T,qc_y.T)
        return(results)

    def tss200_qc(self,method):
        e = ensembl_loci(self.X)
        tss = e.tss200(method=method)
        print("gb mapped dim:",tss.shape)
        self.Xc = to_pc(tss)
        print("pc dim:",self.Xc.shape)
        self.Xc,self.y = map_annotations(self.Xc.T)
        print("go mapped X dims:",self.Xc.shape)
        print("go mapped y dims:",self.y.shape)
        qc_y = self.y.sample(1000)
        #return(self.Xc,qc_y)
        results = self.evaluate(self.Xc.T,qc_y.T)
        return(results)

    def tss5050_qc(self,method):
        e = ensembl_loci(self.X)
        tss = e.tss5050(method=method)
        print("gb mapped dim:",tss.shape)
        self.Xc = to_pc(tss)
        print("pc dim:",self.Xc.shape)
        self.Xc,self.y = map_annotations(self.Xc.T)
        print("go mapped X dims:",self.Xc.shape)
        print("go mapped y dims:",self.y.shape)
        qc_y = self.y.sample(1000)
        #return(self.Xc,qc_y)
        results = self.evaluate(self.Xc.T,qc_y.T)
        return(results)
    
    def gb_qc(self,method):
        e = ensembl_loci(self.X)
        gb = e.gb(method=method)
        print("gb mapped dim:",gb.shape)
        self.Xc = to_pc(gb)
        print("pc dim:",self.Xc.shape)
        self.Xc,self.y = map_annotations(self.Xc.T)
        print("go mapped X dims:",self.Xc.shape)
        print("go mapped y dims:",self.y.shape)
        qc_y = self.y.sample(1000)
        #return(self.Xc,qc_y)
        results = self.evaluate(self.Xc.T,qc_y.T)
        return(results)

    def extended_gb_qc(self,method):
        e = ensembl_loci(self.X)
        gb = e.extended_gb(method=method)
        print("gb mapped dim:",gb.shape)
        self.Xc = to_pc(gb)
        print("pc dim:",self.Xc.shape)
        self.Xc,self.y = map_annotations(self.Xc.T)
        print("go mapped X dims:",self.Xc.shape)
        print("go mapped y dims:",self.y.shape)
        qc_y = self.y.sample(1000)
        #return(self.Xc,qc_y)
        results = self.evaluate(self.Xc.T,qc_y.T)
        return(results)
    
    def expression_qc(self):
        Xe = load_expression_debug()
        Xe = to_pc(Xe.T)
        Xe,y = map_annotations(Xe.T)
        print(Xe.shape,y.shape)
        qc_y = y.sample(1000)
        results = self.evaluate(Xe.T,qc_y.T)
        return(results)
    
    def combined_qc(self):
        e = ensembl_loci(self.X)
        gb = e.extended_gb(method="mean")
        print("gb mapped dim:",gb.shape)
        gb = to_pc(gb)
        print("gbpc:",gb.shape)
        Xe = load_expression_debug()
        Xe = to_pc(Xe.T)
        print("xe:",Xe.shape)
        X = pd.concat([gb.iloc[:,:10],Xe.iloc[:,:10]],axis=1).dropna()
        print("combined:",X.shape)
        X,y = map_annotations(Xe.T)
        qc_y = y.sample(1000)
        results = self.evaluate(X.T,qc_y.T)
        return(results)
        #get gb-mean for meth to pc
    
    def qc(self):
        qc_results = dict()
        #for method in ["mean","median","chalm"]
        #method="mean"
        #method="chalm"
        methods = ["mean","median","chalm2"]
        for method in methods:
            qc_results["tss20002000-%s"%(method)] = self.tss20002000_qc(method)#.metrics()
            qc_results["tss1500-%s"%(method)] = self.tss1500_qc(method)#.metrics()
            qc_results["tss200-%s"%(method)] = self.tss200_qc(method)#.metrics()
            qc_results["tss5050-%s"%(method)] = self.tss5050_qc(method)#.metrics()
            qc_results["gb-%s"%(method)] = self.gb_qc(method)#.metrics()
            qc_results["extended_gb-%s"%(method)] = self.extended_gb_qc(method)#.metrics()
        return(qc_results)
    
    def evaluate(self,X,y):
        model = sklearn.linear_model.LogisticRegression(max_iter=1000)
        p = metalearn.MultilabelProblem(X,y)
        results = p.cross_validate(model)
        return(results)

class tissue_fp(object):
    def __init__(self,X):
        meta = "/data/ncbi.bak/geo/GPL13534/meta/meth_tools/mana/mana_GEO.full_labels"
        meta = pd.read_csv(meta,sep=",",header=0,index_col=0)
        self.meta = meta
        self.X = X 
        print(self.X.shape,self.meta.shape)
    
    def per_tissue_qc(self):
        results = dict()
        for tissue in self.meta.TissueName.unique():
            hits = self.meta[self.meta.TissueName==tissue]
            print(tissue,hits.shape)
            X = self.X[hits.index]
            print(X.shape)
            fp = function_prediction(X)
            result = fp.qc()
            results[tissue] = result
        return(results)
