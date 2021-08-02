#!/usr/bin/env python
# coding: utf-8
# Funciones auxiliares
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import boxcox
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
if not(os.path.isdir("tarea")):
    mkdir("tarea")
def dataScaled(df):
    cat = [var for var in df.columns if not(np.issubdtype(df[var].dtype,np.number))]
    num = df.drop(cat,axis=1)
    # Creating dictionary to store the different data frames
    data = {"original":df}
    # Standarizing data to have mean 0 and variance 1
    scaler = pr.StandardScaler()
    scaler.fit(num)
    data["standarized"] = pd.DataFrame(scaler.transform(num),index=num.index,columns=num.columns)
    data["standarized"][cat] = df[cat]
    data["standarized"] = data["standarized"][df.columns]
    # Centering data to have variance 1 
    scaler = pr.StandardScaler(with_mean=False)
    scaler.fit(num)
    data["withmean"] = pd.DataFrame(scaler.transform(num),index=num.index,columns=num.columns)
    data["withmean"][cat] = df[cat]
    data["withmean"] = data["withmean"][df.columns]
    return data

def boxcoxLambdaTable(df,resCol,alpha=0.05):
    names = []
    lambdas = []
    intervalsBot = []
    intervalsTop = []
    for col in df.columns:
        if col != resCol and np.issubdtype(df[col].dtype,np.number):
            if (df[col]>0).prod():
                names.append(col)
                bx = boxcox(df[col],alpha=alpha)
                lambdas.append(bx[1])
                intervalsBot.append(bx[2][0])
                intervalsTop.append(bx[2][1])
            else:
                print("Can't convert column {0}: not entirely positive".format(col) )
    fin = pd.DataFrame.from_dict({"lambda":lambdas,"Lower confidence interval, alpha = {0}".format(alpha):intervalsBot,"Upper confidence interval, alpha = {0}".format(alpha):intervalsTop})
    fin.index = names
    return fin.transpose()

def bootstrap(df):
    return resample(df,n_samples=df.shape[0]).reset_index(drop=True)

def KFold_strat(X,y,**kwargs):
    splitter = ms.StratifiedKFold(**kwargs)
    iterator = splitter.split(X,y)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in iterator:
        X_train.append(X.iloc[train_index,:])
        y_train.append(y.iloc[train_index])
        X_test.append(X.iloc[test_index,:])
        y_test.append(y.iloc[test_index])
    return X_train,X_test,y_train,y_test

def KFold(X,y,**kwargs):
    splitter = ms.KFold(**kwargs)
    iterator = splitter.split(X,y)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in iterator:
        X_train.append(X.iloc[train_index,:])
        y_train.append(y.iloc[train_index])
        X_test.append(X.iloc[test_index,:])
        y_test.append(y.iloc[test_index])
    return X_train,X_test,y_train,y_test

def apparentErrorRate(mod,df,resCol):
    X = df.drop(resCol,axis=1)
    y = df[resCol]
    fit = mod.fit(X,y)
    res = fit.predict(X)
    fin = np.mean(y != res)
    classes = list(set(y))
    perClass = np.zeros(len(classes))
    for j,val in enumerate(classes):
        curr = X[y==val]
        res = fit.predict(curr)
        perClass[j] = np.mean(y[y==val] != res)
    final = [fin]
    final.extend(perClass)
    return final,[0 for f in final]

def trainTestErrorRates(mod,df,resCol,n=100,size=0.5,equalRatios=False):
    X = df.drop(resCol,axis=1)
    y = df[resCol]
    fin = np.zeros(n)
    classes = list(set(y))
    perClass = [np.zeros(n) for val in classes]
    for i in range(n):
        if size < 1.0 and size > 0:
            if equalRatios:
                X_train, X_test, y_train, y_test = ms.train_test_split(X,y,train_size=size,stratify=y)
            else:
                X_train, X_test, y_train, y_test = ms.train_test_split(X,y,train_size=size)
            fit = mod.fit(X_train,y_train)
            res = fit.predict(X_test)
            fin[i] = np.mean(y_test != res)
            for j,val in enumerate(classes):
                curr = X[y==val]
                res = fit.predict(curr)
                perClass[j][i] = np.mean(y[y==val] != res)
        else:
            raise ValueError("Size {0} is not in (0,1)".format(size))
    final1 = [np.mean(fin)]
    final2 = [np.std(fin)] 
    for cla in perClass:
        final1.append(np.mean(cla))
        final2.append(np.std(cla))
    return final1, final2

def bootstrapErrorRate(mod,df,resCol,n=100):
    fin = []
    for i in range(n):
        newdf = bootstrap(df)
        errors = apparentErrorRate(mod,newdf,resCol)
        fin.append(errors)
    fin = np.transpose(fin)
    final1 = [np.mean(f) for f in fin]
    final2 = [np.std(f) for f in fin]
    return final1 , final2

def crossValidationErrorRate(mod,df,resCol,k=2,equalRatios=True,n=100):
    errors = np.zeros(n)
    classes = list(set(df[resCol]))
    perClass = [np.zeros(n) for val in classes]
    if equalRatios:
        splitFunc = KFold_strat
    else:
        splitFunc = KFold
    for i in range(n):
        X_train,X_test,y_train,y_test = splitFunc(df.drop(resCol,axis=1),df[resCol],n_splits=k,shuffle=True)
        temps = np.zeros(k)
        classTemps = [np.zeros(k) for val in classes]
        for j in range(k):
            fit = mod.fit(X_train[j],y_train[j])
            res = fit.predict(X_test[j])
            temps[j] = np.mean(y_test[j] != res)
            for l,val in enumerate(classes):
                curr = X_test[j][y_test[j]==val]
                res = fit.predict(curr)
                classTemps[l][j] = np.mean(y_test[j][y_test[j]==val] != res)
        errors[i] = np.mean(temps)
        for l,val in enumerate(classes):
            perClass[l][i] = np.mean(classTemps[l])
    final1 = [np.mean(errors)]
    final2 = [np.std(errors)]
    final1.extend([np.mean(v) for v in perClass])
    final2.extend([np.std(v) for v in perClass])
    return final1, final2

def resamplingComparison(model,df,resCol,k=5,n=100,size=0.5,equalRatios = True,stds=False):
    classes = list(set(df[resCol]))
    names = ["Normal","Bootstrap","Training/Test, fraction = {0}".format(size),"Cross validation, k = {0}".format(k)]
    errors = [apparentErrorRate(model,df,resCol), 
              bootstrapErrorRate(model,df,resCol,n=n), 
              trainTestErrorRates(model,df,resCol,n=n,size=size,equalRatios=equalRatios),
              crossValidationErrorRate(model,df,resCol,k=k,equalRatios=equalRatios,n=n)]
    cols = ["Global"]
    cols += ["Class {0}".format(c) for c in classes]
    cols += ["Global STD"]
    cols += ["Class {0} STD".format(c) for c in classes]
    res = pd.DataFrame(columns = cols)
    for i,tab in enumerate(errors):
        res.loc[i] = tab[0] + tab[1]
    res["method"] = names
    res = res[np.roll(res.columns.to_list(),1)]
    if not(stds):
        res = res.iloc[:,range(res.shape[1])[:-(len(classes)+1)]]
    return res

def modelComparison(models,dfs,resCol,errorFunc,names=[],stds=False):
    classes = list(set(df[resCol]))
    if type(dfs)!= list:
        dfs = [dfs for m in models]
    if not bool(names):
        names = [str(mod).split("(")[0] for mod in models]
    elif len(names) != len(models) or len(dfs)!=len(models):
        raise ValueError("length of names, models and dfs do not match")
    cols = ["Global"]
    cols += ["Class {0}".format(c) for c in classes]
    cols += ["Global STD"]
    cols += ["Class {0} STD".format(c) for c in classes]
    res = pd.DataFrame(columns = cols)
    for i,mod in enumerate(models):
        tab = errorFunc(mod,dfs[i],resCol)
        res.loc[i] = tab[0] + tab[1]
    res["Model"] = names
    res = res[np.roll(res.columns.to_list(),1)]
    if not(stds):
        res = res.iloc[:,range(res.shape[1])[:-(len(classes)+1)]]
    return res
    

# Problema 1
df = pd.read_csv("pimate.csv")
df = df.append(pd.read_csv("pimatr.csv"),ignore_index=True)
print(df.head())

data = dataScaled(df)

models = [
    LinearDiscriminantAnalysis(),
    GaussianNB(),
    LogisticRegression(dual=False,max_iter=10**6),
    SVC()
]

df1 = data["original"].copy()
df1["ped*age"] = df1["ped"]*df1["age"]
df2 = data["original"].copy()
df2["ped*bp"] = df2["ped"]*df2["bp"]
df3 = data["original"][["glu","bmi","ped","age","type"]].copy()
df3["age^2"] = df3["age"]*df3["age"]
df4 = data["original"].copy()
dfs = [df1,df2,df3,df4]

tables = [resamplingComparison(models[i],dfs[i],"type",n=5) for i in range(len(models))]
tables

n = 500
names = ["Analisis de Discriminante Lineal","Naive Bayes","Regresión Logística","Support Vector Machine"]
errors = [apparentErrorRate,lambda model,df,resCol: bootstrapErrorRate(model,df,resCol,n=n), 
              lambda model,df,resCol : trainTestErrorRates(model,df,resCol,n=n,size=0.75,equalRatios=True),
              lambda model,df,resCol : crossValidationErrorRate(model,df,resCol,k=5,equalRatios=True,n=50)]

res = []
for err in errors[:-1]:
    print(str(err))
    res.append(modelComparison(models,dfs,"type",err,names=names))

res

tab = "p{3cm}"
for col in res[0].columns:
    tab += "|c"
res[0].to_latex(buf=os.path.join("tarea","41-apparent.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[1].to_latex(buf=os.path.join("tarea","41-boot.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[2].to_latex(buf=os.path.join("tarea","41-traintest.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)

# Problema 2
df = pd.read_csv("cad1.csv",index_col=0)
resCol="CAD"
dfCoded = df.copy()
for col in df.columns: 
    if col!=resCol and df[col].dtype==np.dtype("O"):
        dfCoded[col] = df[col].astype("category").cat.codes
print(dfCoded.head())

models = [
    GaussianNB(),
    LogisticRegression(dual=False,max_iter=10**6),
    SVC()
]

df1 = dfCoded.copy()
df1["Sex*AMI"] = df1["Sex"]*df1["AMI"]
df2 = dfCoded[["AngPec","AMI","STcode","STchange","Hyperchol","CAD"]].copy()
df3 = dfCoded.copy()
dfs = [df1,df2,df3]

n = 500
names = ["Naive Bayes","Regresión Logística","Support Vector Machine"]
errors = [apparentErrorRate,
              lambda model,df,resCol : trainTestErrorRates(model,df,resCol,n=n,size=0.75,equalRatios=True),
              lambda model,df,resCol : crossValidationErrorRate(model,df,resCol,k=5,equalRatios=True,n=n)]

res = []
for err in errors:
    print(str(err))
    res.append(modelComparison(models,dfs,"CAD",err,names=names))

tab = "p{3cm}"
for col in res[0].columns:
    tab += "|c"
res[0].to_latex(buf=os.path.join("tarea","42-apparent.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[1].to_latex(buf=os.path.join("tarea","42-traintest.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[2].to_latex(buf=os.path.join("tarea","42-crossval.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)

# Problema 3
df = pd.read_csv("Glucose1.txt",index_col="Patient")
#df["Class"] = df["Class"].astype("O")
data = dataScaled(df)
print(data["original"].head())

modpredict = data["original"][["InsulinResp","Class"]].copy()
interactions = ["Fglucose*InsulinResp","GlucoseInt*InsulinResp"]
for inter in interactions:
    columns = inter.split("*")
    modpredict[inter] = data["original"][columns].product(axis=1)

n = 500
names = ["Regresión logística"]
errors = [apparentErrorRate,lambda model,df,resCol: bootstrapErrorRate(model,df,resCol,n=n), 
              lambda model,df,resCol : trainTestErrorRates(model,df,resCol,n=n,size=0.75,equalRatios=True),
              lambda model,df,resCol : crossValidationErrorRate(model,df,resCol,k=5,equalRatios=True,n=n)]

res = []
for err in errors:
    print(str(err))
    res.append(modelComparison([LogisticRegression(dual=False,max_iter=10**6)],[modpredict],"Class",err,names=names))

tab = "p{3cm}"
for col in res[0].columns:
    tab += "|c"
res[0].to_latex(buf=os.path.join("tarea","43-apparent.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[1].to_latex(buf=os.path.join("tarea","43-boot.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[2].to_latex(buf=os.path.join("tarea","43-traintest.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)
res[3].to_latex(buf=os.path.join("tarea","43-crossval.tex"),float_format="{:0.4f}".format,index=False,column_format=tab)



