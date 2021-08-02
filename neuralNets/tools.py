import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def dataScaler(df):
    cat = [var for var in df.columns if not(np.issubdtype(df[var].dtype,np.number))]
    num = df.drop(cat,axis=1)
    # Standarizing data to have mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(num)
    data = pd.DataFrame(scaler.transform(num),index=num.index,columns=num.columns)
    data[cat] = df[cat]
    data = data[df.columns]
    return data

def kfold(X,y,**kwargs):
    splitter = KFold(**kwargs)
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

def kfold_strat(X,y,**kwargs):
    splitter = StratifiedKFold(**kwargs)
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
 

def saveModel(model,folderName,historyName=""):
    if not(os.path.isdir(folderName)):
        os.mkdir(folderName)
    model.save(folderName)
    if bool(historyName):
        df = pd.DataFrame.from_dict(model.history.history)
        df.insert(loc=0,column="epochs", value = model.history.epoch)
        df.to_csv(os.path.join(folderName,"{0}.csv".format(historyName)),index=False)

def readModel(folderName,historyName="",model=True):
    if not(os.path.isdir(folderName)):
        raise ValueError("No folder named {0}".format(folderName))
    if model:
        try:
            newmod = tf.keras.models.load_model(folderName)
        except KeyError:
            newmod = tf.keras.models.load_model(os.path.join(folderName,"..","model.h5"),custom_objects={"swish":tf.nn.swish})
        if bool(historyName):
            df = pd.read_csv(os.path.join(folderName,"{0}.csv".format(historyName)))
            return newmod, df
        else:
            return newmod, pd.DataFrame()
    else: 
        if bool(historyName):
            df = pd.read_csv(os.path.join(folderName,"{0}.csv".format(historyName)))
            return df
        else:
            return pd.DataFrame()
    
def readCrossValidationModel(folderName,model):
    folders = [name for name in os.listdir(folderName) if os.path.isdir(os.path.join(folderName,name)) and not("." in name)]
    print("Sub folders found: {0}".format(folders))
    subs = [name for name in folders if name[0:4]!="main"]
    aux = []
    print("Reading main model")
    aux.append(readModel(os.path.join(folderName,"main"),historyName="trainingStats",model=model))
    for sub in subs:
        print("Reading {0}".format(sub))
        aux.append(readModel(os.path.join(folderName,sub),historyName="trainingStats",model=model))
    if model:
        models = [i for i,j in aux]
        results = [j for i,j in aux]
        del aux
        return models, results
    else:
        return aux


def getPredictions(data,model):
    # getting size of data
    #if batchSize == 0:
    #    batchSize =  data.reduce(0,lambda x,_: x+1).numpy()
    labels = model.predict(data)
    labels = np.argmax(tf.nn.softmax(labels),1)
    return labels

def getPredictionProbs(data,model,index):
    # getting size of data
    #if batchSize == 0:
    #    batchSize =  data.reduce(0,lambda x,_: x+1).numpy()
    labels = model.predict(data)
    labels = tf.nn.softmax(labels).numpy()
    return labels[index]

def getSinglePredictions(data,model,index):
    # getting size of data
    #if batchSize == 0:
    #    batchSize =  ata.reduce(0,lambda x,_: x+1).numpy()
    labels = model.predict(np.array([data[index]]))
    labels = np.argmax(tf.nn.softmax(labels),1)
    return labels[0]

def getWrongPredictions(data,model,y_true):
    labels = getPredictions(data,model)
    return np.where(y_true!=labels)[0]
    #return [index for index,val in enumerate(y_true) if labels[index] != val]


def accuracyRate(y_true,y_pred):
    return 1-np.mean((y_pred - y_true) != 0)

def perClassAccuracyRate(y_true,y_pred):
    # assumes classes are already coded as integers from 0 to nclass-1
    classes = set(y_true)
    n_classes = len(classes)
    errors = np.zeros(n_classes)
    elements = np.ones(n_classes)
    for i,y in enumerate(y_pred):
        elements[y] += 1
        if y!=y_true[i]:
            errors[y] += 1
    return [1-errors[i]/elements[i] for i in range(n_classes)]

def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n

def ginicTF(actual:tf.Tensor,pred:tf.Tensor):
    n = int(actual.get_shape()[-1])
    inds =  tf.reverse(tf.nn.top_k(pred,n)[1],axis=[0]) # this is the equivalent of np.argsort
    a_s = tf.gather(actual,inds) # this is the equivalent of numpy indexing
    a_c = tf.cumsum(a_s)
    giniSum = tf.reduce_sum(a_c)/tf.reduce_sum(a_s) - (n+1)/2.0
    return giniSum / n

def basicStatistics(data,y_true,model):
        stats = pd.DataFrame(model.history.history)
        y_pred = getPredictions(data,model)
        stats["accuracyRate"] = [accuracyRate(y_true,y_pred)]
        return stats
    
def classStatistics(data,y_true,model):
        stats = pd.DataFrame(model.history.history)
        y_pred = getPredictions(data,model)
        stats["accuracyRate"] = [accuracyRate(y_true,y_pred)]
        perClass = perClassAccuracyRate(y_true,y_pred)
        for index,val in enumerate(perClass):
            stats["perClassAccuracyRate{0}".format(index)] = [val]
        return stats
    
def customStatistics(data,y_true,model,funcList):
    stats = pd.DataFrame(model.history.history)
    y_pred = getPredictions(data,model)
    for func in funcList:
        stat = func(y_true,y_pred)
        if isinstance(stat,list):
            for index,val in enumerate(stat):
                stats["{0}{1}".format(func.__name__,index)] = [val]
        else:
            stats["{0}".format(func.__name__)] = [stat]
    return stats

def labelData(data,y_test_list):
    original = pd.DataFrame(index = data.index)
    for i,y in enumerate(y_test_list):
        original.loc[y.index,"foldTestClass"]  = i+1
    original["foldTestClass"] = original["foldTestClass"].astype("int64")
    return original
       