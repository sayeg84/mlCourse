import sklearn.cluster as clus
import sklearn.decomposition as dc
import sklearn.preprocessing as pr
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
if not(os.path.isdir("tarea")):
    os.mkdir("tarea")
# function to scale data acording to different
def dataScaled(data):
    # Creating dictionary to store the different data frames
    data = {"original":df}
    # Standarizing data to have mean 0 and variance 1
    scaler = pr.StandardScaler()
    scaler.fit(df)
    data["standarized"] = pd.DataFrame(scaler.transform(df),index=df.index,columns=df.columns)
    # Centering data to have variance 1 
    scaler = pr.StandardScaler(with_mean=False)
    scaler.fit(df)
    data["withmean"] = pd.DataFrame(scaler.transform(df),index=df.index,columns=df.columns)
    return data
# function to export the PCA coefficient table
def compCoefTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    return pd.DataFrame(model.components_, index = names,columns = data.columns)

# function to export the singular value table
def singvalTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    return pd.DataFrame(model.singular_values_,index =names).transpose()

# function to create the PVE table using the scikit-learn package
def pveTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    return pd.DataFrame(model.explained_variance_ratio_,index=names,columns = ["PVE"]).transpose()

#function to calculate the PVE of the m-th principal component
def pve(m,model,data):
    n,p = df.shape
    norm = 0
    s = 0
    for i in range(n):
        temp = 0
        for j in range(p):
            #print("s = {0}".format(s))
            #print("norm = {0}".format(norm))
            temp = temp + model.components_[m][j]*data.iloc[i][j]
            norm = norm + data.iloc[i][j]**2
        s = s + temp**2
    return s/norm

#function to create the PVE table using the PVE(model,data) function
def pveFromFormulaTable(model,data):
    p = len(data.columns)
    c = int(np.floor(np.log10(p)+1))
    names = ["PC"+"{0}".format(i).zfill(c) for i in range(len(data.columns))]
    vals = [pve(i,model,data) for i in range(p)]
    return pd.DataFrame(vals,index=names,columns = ["PVE"]).transpose()

# Auxiliar function taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# for plotting dendogram of agglomerative clustering
def plotDendogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# function for returning lists with all clusters 
def clusters(fit,data):
    clu = [[] for i in range(fit.n_clusters_)]
    for i in range(fit.n_leaves_):
        clu[fit.labels_[i]].append(data.index[i])
    return clu

def plotClusteringResults(fit,df,s=20):
    #labels = (fit.labels_-fit.labels_[0]) % nclus
    labels = fit.labels_
    clusColors = ["C{0}".format(i) for i in labels]
    normColors = ["C{0}".format(i) for i in df["initG"]]

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],s=s,c=clusColors)
    ax1.view_init(30, 45)
    ax1.set_title("Clustering")

    ax2.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],s=s,c=normColors)
    ax2.view_init(30, 45)
    ax2.set_title("Normal Groups")

# Exercise 8

# Reading data
df = pd.read_csv("USArrests.csv",header=0,index_col=0)
print(df.head())
df.head().to_latex(buf="tarea/1-head.tex")
# Making PCA over standarized data for effecive PCA

data = dataScaled(df)
print(data["standarized"].head())
model = dc.PCA()
model.fit(data["standarized"])
x = compCoefTable(model,data["standarized"])
print(x)
x.to_latex(buf="tarea/1-coeffTable.tex",float_format = "{:0.4f}".format)
print(singvalTable(model,data["standarized"]))
# calculating PVE table from package
x = pveTable(model,data["standarized"])
print(x)
x.to_latex(buf="tarea/1-pve.tex",float_format = "{:0.4f}".format)
# calculating PVE table using formula
x = pveFromFormulaTable(model,data["standarized"])
print(x)
x.to_latex(buf="tarea/1-pveForm.tex",float_format = "{:0.4f}".format)

# Exercise 9

print(data["withmean"].head())
# Performinc hierarchical clustering over different linkages
res = pd.DataFrame(index=data["original"].index)
for l in ["complete","single","average"]:
    for name, df in data.items():
        model = clus.AgglomerativeClustering(linkage=l,affinity = "euclidean",distance_threshold=0.0,n_clusters=None)
        fit  = model.fit(X=df)
        plt.figure(figsize=(12,5))
        plotDendogram(fit,labels=df.index)
        plt.grid()
        plt.title("Linkage: {0}, Data: {1}".format(l,name))
        plt.tight_layout()
        plt.savefig("tarea/2-{0}{1}.pdf".format(l,name))
        plt.show()
        model = clus.AgglomerativeClustering(linkage=l,affinity = "euclidean",n_clusters=3)
        fit  = model.fit(X=df)
        # making first observation be in cluster 0
        res["Cluster {0} {1}".format(l,name)] = (fit.labels_ - fit.labels_[0]) % 3
tabs = "c"
for i in range(0,res.shape[1]):
    tabs = tabs + "p{1.5cm}"
res.index = [a[0:7] for a in res.index]
res.to_latex(buf = "tarea/2-clusresults.tex",column_format=tabs,longtable=True)

# Exercise 10

n=50
ng = 3
gobs = 20
g = []
df = pd.DataFrame()
for i in range(ng):
    mean = np.zeros(n)
    mean[i] = 1.0
    obs = np.array([np.random.rand(n) + mean for i in range(gobs)])
    aux = pd.DataFrame(obs)
    aux["initG"] = i
    df = df.append(aux,ignore_index=True)
print(df.head())
data = dataScaled(df.drop(columns="initG"))
for name,df1 in data.items():
    df1["initG"] = df["initG"]
model = clus.KMeans(n_clusters=3)
fit = model.fit(X=data["original"].drop(columns="initG"))
fig = plt.figure(figsize=(12,6))
plotClusteringResults(fit,data["original"],s=80)
plt.tight_layout()
plt.savefig("tarea/3-initClus.pdf")
plt.show()
model = dc.PCA()
model.fit(data["original"].drop(columns="initG"))
compCoefTable(model,data["original"].drop(columns="initG"));
trans = model.transform(data["original"].drop(columns="initG"))
trans = pd.DataFrame(trans,columns = pveTable(model,data["original"].drop(columns="initG")).columns)
trans["initG"] = data["original"]["initG"]
normColors = ["C{0}".format(i) for i in data["original"]["initG"]]
plt.figure()
plt.title("Principal components")
plt.xlabel("PC0")
plt.ylabel("PC1")
plt.scatter(trans["PC00"],trans["PC01"], color = normColors)
plt.grid()
plt.tight_layout()
plt.savefig("tarea/3-pca.pdf")
plt.show()
print(pveTable(model,data["original"].drop(columns="initG")))
res = pd.DataFrame(index = data["original"].index)
for name,df in data.items():
    for j in [3,2,4]:
        model = clus.KMeans(n_clusters=j)
        fit = model.fit(X=df.drop(columns="initG"))
        labels = (fit.labels_ - fit.labels_[0] )% j
        res["K: {0} , data: {1}".format(j,name)] = labels
        fig = plt.figure(figsize=(12,6))
        plotClusteringResults(fit,df,s=80)
        plt.suptitle("N clusters = {0}, data: {1}".format(j,name),fontsize=20)
        plt.tight_layout()
        plt.savefig("tarea/3-{1}-{0}.pdf".format(j,name))
        plt.show()
tabs = "c"
for i in range(0,res.shape[1]):
    tabs = tabs + "p{1.5cm}"
res.to_latex(buf = "tarea/3-clusresults.tex",column_format=tabs,longtable=True)
for j in [3,2,4]:
    model = clus.KMeans(n_clusters=j)
    fit = model.fit(X=trans.drop(columns="initG"))
    fig = plt.figure(figsize=(12,6))
    plotClusteringResults(fit,trans,s=50)
    plt.suptitle("N clusters = {0}, data: transformed to PC".format(j,name),fontsize=20)
    plt.tight_layout()
    plt.savefig("tarea/3-fullpcs-{0}.pdf".format(j))
    plt.show()
res = pd.DataFrame(index=data["original"].index)
for j in [3,2,4]:
    model = clus.KMeans(n_clusters=j)
    fit = model.fit(X=trans.iloc[:,[0,1]])
    labels = (fit.labels_ - fit.labels_[0]) % j
    res["data: pca 2 vars, K={0}".format(j)] = labels
    fig = plt.figure(figsize=(12,6))
    plotClusteringResults(fit,trans,s=50)
    plt.suptitle("N clusters = {0}, data: transformed to PC".format(j,name),fontsize=20)
    plt.tight_layout()
    plt.savefig("tarea/3-2pcs-{0}.pdf".format(j))
    plt.show()
tabs = "c"
for i in range(0,res.shape[1]):
    tabs = tabs + "p{3cm}"

res.to_latex(buf = "tarea/3-clusresults-pca.tex",column_format=tabs,longtable=True)

# Exercise 11

# Reading Data
df = pd.read_csv("Ch10Ex11.csv",header=None)
df = df.transpose()
data = dataScaled(df)
print(df.head().iloc[:,0:5])
df.head().iloc[:,0:5].to_latex("tarea/4-initData.tex",float_format="{:0.4f}".format)
res = pd.DataFrame(index=data["original"].index)
for l in ["complete","single","average"]:
    for name, df in data.items():
        model = clus.AgglomerativeClustering(linkage=l,affinity = "correlation",distance_threshold=0.0,n_clusters=None)
        fit  = model.fit(X=df)
        plt.figure(figsize=(12,5))
        plotDendogram(fit,labels=df.index)
        plt.grid()
        plt.title("Linkage: {0}, Data: {1}".format(l,name))
        plt.tight_layout()
        plt.savefig("tarea/4-{0}{1}.pdf".format(l,name))
        plt.show()
        model = clus.AgglomerativeClustering(linkage=l,affinity = "correlation",n_clusters=2)
        fit  = model.fit(X=df)
        # making first observation be in cluster 0
        res["Cluster {0} {1}".format(l,name)] = (fit.labels_ - fit.labels_[0]) % 2
tabs = "c"
for i in range(0,res.shape[1]):
    tabs = tabs + "p{1.5cm}"
res.to_latex(buf="tarea/4-clusres.tex",longtable=True,column_format=tabs)
# method 1: calculating variance and getting gen that maximizes it
df = data["standarized"]
genDif = {}
for name,df in data.items():
    health = df.iloc[0:20,:]
    sick = df.iloc[20:,:]
    norms = []
    for i in data["original"].columns:
        diffvec = [health.loc[n,i] - sick.loc[m,i] for n in health.index for m in sick.index]
        norms.append(np.linalg.norm(diffvec))
    genDif[name] = norms.index(max(norms))
print(genDif)
for name,gen in genDif.items():
    plt.figure()
    plt.title("Gen that maximizes |A_k|: {0} \n Linkage: {1}".format(gen,name))
    plt.hist(health.loc[:,gen],alpha=0.7)
    plt.hist(sick.loc[:,gen],alpha=0.7)
    plt.grid()
    plt.tight_layout()
    plt.savefig("tarea/4-Ak-{0}.pdf".format(name))
    plt.show()
# Method 1: calculating variance and getting 20 gens that maximize it
genDif = {}
dists = {}
n=20
for name,df in data.items():
    health = df.iloc[0:20,:]
    sick = df.iloc[20:,:]
    norms = []
    for i in data["original"].columns:
        diffvec = [health.loc[n,i] - sick.loc[m,i] for n in health.index for m in sick.index]
        norms.append(np.linalg.norm(diffvec))
    genDif[name] = list(np.argsort(norms))[-n:]
    dists[name] = list(sorted(norms))[-n:]
print(genDif)
res = pd.DataFrame()
for name,lis in genDif.items():
    res[name] = lis
    res[name + " variance distance"] = dists[name]
tabs = "c"
for i in range(res.shape[1]):
    tabs += "p{2cm}"
res.to_latex("tarea/4-variance-distance.tex",column_format=tabs,longtable=True,float_format="{:0.2f}".format)
# Method 2: looking for a gen which removing it causes clustering fail
df = data["standarized"]
for l in ["complete","single","average"]:
    for i in df.columns:
        aux = df.drop(columns=i)
        model = clus.AgglomerativeClustering(linkage=l,affinity = "correlation",n_clusters=2)
        fit  = model.fit(X=aux)
        cluslabels = [0 for i in range(20)] + [1 for i in range(20)]
        labels = list((fit.labels_ - fit.labels_[0] ) % 2)
        if cluslabels != labels:
            print("Whitout this gen, clustering fails")
            print(cluslabels)
            print(labels)
            print(i)
# Looking for the gene that makes posible the clustering with only with him as variable
genDif = {}
for l in ["complete","single","average"]:
    for i in df.columns:
        aux = pd.DataFrame(df[i])
        # distance has to be euclidean in order to avoid problems
        model = clus.AgglomerativeClustering(linkage=l,affinity = "euclidean",n_clusters=2)
        fit = model.fit(X = aux)
        cluslabels = [0 for i in range(20)] + [1 for i in range(20)]
        labels = list((fit.labels_ - fit.labels_[0] ) % 2)
        if cluslabels == labels:
            print("With this gene, clustering works ")
            print(i)
# Method 3: Looking for the gene that minimizes the classification error
genDif = {}
for l in ["complete","single","average"]:
    vec = []
    for i in df.columns:
        aux = pd.DataFrame(df[i])
        # distance has to be euclidean in order to avoid problems
        model = clus.AgglomerativeClustering(linkage=l,affinity = "euclidean",n_clusters=2)
        fit = model.fit(X = aux)
        cluslabels = [0 for i in range(20)] + [1 for i in range(20)]
        labels = list((fit.labels_ - fit.labels_[0] ) % 2)
        errors = 0
        for i,lab in enumerate(cluslabels):
            if labels[i] != lab:
                errors = errors + 1
        vec.append(errors/df.shape[0])
    mingens = [x[0] for x in enumerate(vec) if vec[x[0]]==min(vec)]
    if len(mingens) == 1:
        genDif[l] = mingens[0]
    else:
        print("These gens minimize error:")
        print(mingens)
print(genDif)
for name,gen in genDif.items():
    plt.figure()
    plt.title("Gen with minimal classification error: {0} \n Linkage: {1}".format(gen,name))
    plt.hist(health.loc[:,gen],alpha=0.7)
    plt.hist(sick.loc[:,gen],alpha=0.7)
    plt.grid()
    plt.tight_layout()
    plt.savefig("tarea/4-minimal-{0}.pdf".format(name))
    plt.show()
# Looking for the 20 genes that minimizes the classification error
genDif = {}
errs = {}
n = 20
for l in ["complete","single","average"]:
    vec = []
    for i in df.columns:
        aux = pd.DataFrame(df[i])
        # distance has to be euclidean in order to avoid problems
        model = clus.AgglomerativeClustering(linkage=l,affinity = "euclidean",n_clusters=2)
        fit = model.fit(X = aux)
        cluslabels = [0 for i in range(20)] + [1 for i in range(20)]
        labels = list((fit.labels_ - fit.labels_[0] ) % 2)
        errors = 0
        for i,lab in enumerate(cluslabels):
            if labels[i] != lab:
                errors = errors + 1
        vec.append(errors/df.shape[0])
    mingens = list(np.argsort(vec)[0:n])
    genDif[l] = mingens
    errs[l] = sorted(vec)[0:n]
print(genDif)
res = pd.DataFrame()
for name,lis in genDif.items():
    res[name] = lis
    res[name + " error ratio"] = errs[name]
tabs = "c"
for i in range(res.shape[1]):
    tabs += "p{2cm}"
res.to_latex("tarea/4-errors-ratio.tex",column_format=tabs,longtable=True)
res = pd.DataFrame(index = data["original"].index)


