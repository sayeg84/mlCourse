

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import time
import random
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler


# fixating random seed for reproduction 

seed = 1
os.environ['PYTHONHASHSEED']=str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


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

def readModel(folderName,historyName=""):
    if not(os.path.isdir(folderName)):
        raise ValueError("No folder named {0}".format(folderName))
    newmod = tf.keras.models.load_model(folderName)
    if bool(historyName):
        df = pd.read_csv(os.path.join(folderName,"{0}.csv".format(historyName)))
        return newmod, df
    else:
        return newmod, pd.DataFrame()

def getPredictions(data,model,batchSize = 0):
    # getting size of data
    #if batchSize == 0:
    #    batchSize =  data.reduce(0,lambda x,_: x+1).numpy()
    labels = model.predict(data)
    labels = np.argmax(tf.nn.softmax(labels),1)
    return labels

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
       



print()                                                            
print("reading Data")

test = pd.read_csv("MNISTtest.csv")
train = pd.read_csv("MNISTtrain.csv")
validation = pd.read_csv("MNISTvalidate.csv")
names = {"C785":"Class"}
test = test.rename(columns=names)
train = train.rename(columns=names)
validation = validation.rename(columns=names)

x_test_dataset=  dataScaler(test.drop("Class",axis=1)).values
y_test_dataset = test["Class"].values
x_train_dataset=  dataScaler(train.drop("Class",axis=1)).values
y_train_dataset = train["Class"].values
validation_dataset = dataScaler(validation).values


print()
print("Done")
print()



print()
print("Initializing TensorFlow")
print()

# searching for GPU
print()
print("Available devices:")
print()
print(tf.config.list_physical_devices())
# intialization
xs = np.linspace(-2,2,1)
tf.keras.activations.elu(xs)


print()
print("Done")
print()

# regular models 

def oneLayer1():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1")(inputs)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def oneLayer2(activation="relu",penalization=0.0, size=100):
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    x = tf.keras.layers.Dense(size,activation=activation,name="Dense_Layer_1",kernel_regularizer=tf.keras.regularizers.l1(l=penalization))(inputs)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def twoLayers(activation=["relu","relu"],penalization=[0.0,0.0],size=[100,100]):
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    x = tf.keras.layers.Dense(size[0],activation="relu",name="Dense_Layer_1")(inputs)
    x = tf.keras.layers.Dense(size[1],activation="relu",name="Dense_Layer_2")(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def twoLayersDropout(activation=["relu","relu"],penalization=[0.0,0.0],size=[100,100],dropout=0.5):
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    x = tf.keras.layers.Dense(size[0],activation="relu",name="Dense_Layer_1")(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(size[1],activation="relu",name="Dense_Layer_2")(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def fourLayers(activation = ["relu"]):
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1")(inputs)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_2")(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_3")(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_4")(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def normalLayered(activation,penalization,size,dropout,input_dropout=0.0,layerNames=[]):
    nLayers = len(activation)
    if not bool(layerNames):
        nZeros = int(np.floor(np.log10(nLayers))) + 1
        layerNames = ["Dense_layer_{0}".format(str(i+1).zfill(nZeros)) for i in range(nLayers)]
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    x = tf.keras.layers.Dropout(input_dropout)(inputs)
    x = tf.keras.layers.Dense(size[0],activation=activation[0],name=layerNames[0],kernel_regularizer=tf.keras.regularizers.l1(l=penalization[0]))(x)
    x = tf.keras.layers.Dropout(dropout[0])(x)
    for i in range(len(activation)-1):
        x = tf.keras.layers.Dense(size[i+1],activation=activation[i+1],name=layerNames[i+1],kernel_regularizer=tf.keras.regularizers.l1(l=penalization[i+1]))(x)
        x = tf.keras.layers.Dropout(dropout[i+1])(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")


def unevenLayeredModel(layers  = 3 , l = 0.0, mode = "lin"):
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    if mode == "lin":
        layList = [int(np.ceil(x)) for x in np.linspace(1,28*28,layers+2)]
    elif mode =="log":
        layList = [int(np.ceil(x)) for x in np.logspace(1,2*np.log10(28),layers+2)]
    else:
        raise ValueError("mode {0} not supported".format(mode))
    layList.reverse()
    x = tf.keras.layers.Dense(layList[1],activation="relu",name="Dense_Layer_1",kernel_regularizer=tf.keras.regularizers.l1(l=l))(inputs)
    for i,lay in enumerate(layList[2:-1]):
        x = tf.keras.layers.Dense(lay,activation="relu",name="Dense_Layer_{0}".format(i+2),kernel_regularizer=tf.keras.regularizers.l1(l=l))(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="unevenlayered")

# convoluted models

def convMod1():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1]),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv1")

def convMod2():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1]),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv2")

def convMod3():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1]),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv3")

def convMod4():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1]),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv4")

def convMod5():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1]),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv5")

def convMod6():
    inputs = tf.keras.Input(shape=(28,28,1),name="Input")
    #x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform',input_shape=(28,28,1))(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv6")

# logistic regression model

def logReg():
    inputs = tf.keras.Input(shape=(test.drop("Class",axis=1).shape[1],),name="Input")
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(inputs)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")


def basicCompile(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

def interestingCompile(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy',
              tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])


def testSplitting(n_folds):
    histBase = np.histogram(train["Class"],bins=10)[0]
    histBase  = 1/sum(histBase)*histBase
    X_train,X_test,y_train,y_test = kfold_strat(train.drop("Class",axis=1),train["Class"],n_splits=n_folds,shuffle=True)
    print("Original proportions:")
    print(histBase)
    for i in range(n_folds):
        histNew = np.histogram(y_train,bins=10)[0]
        histNew  = 1/sum(histNew)*histNew
        print("fold {0} proportions: ".format(i+1))
        print(histNew)

def testTrainModel(x_train_data,y_train_data,x_test_data,y_test_data,y_true,modName,model,compilationFunc,n_epochs,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    print(saveFreq)
    
    compilationFunc(model)
    result = pd.DataFrame() 
    initTime = time.perf_counter()
    for i in range(n_epochs):
        with tf.device(deviceid):
            print("{1}, epoch {0}".format(i+1,modName))
            model.fit(x_train_data,y_train_data,
                        batch_size = batch_size,
                      epochs=1,
                      validation_data=(x_test_data,y_test_data))                       
            stats = statsFunc(x_train_data,y_true,model)
            result = result.append(stats,ignore_index=True)
        if i % saveFreq == 0:
            if not(os.path.isdir(modFunc.__name__)):
                os.mkdir(modFunc.__name__)
            # saving main model
            saveModel(model,os.path.join(modFunc.__name__,modName))
            result.to_csv(os.path.join(modFunc.__name__,modName,"trainingStats.csv"),index=False)
            # saving predictions made from validation data
            y_valid = getPredictions(validation_dataset,model)
            np.savetxt(os.path.join(modFunc.__name__,modName,"predictions.csv"),y_valid,delimiter=",",fmt="%i" )
            # saving MetaParams
            pd.DataFrame.from_dict({"time":[time.perf_counter()-initTime],"batch size":[batch_size]}).to_csv(os.path.join(modFunc.__name__,"metaParams.csv"))
    del model, result
        
def testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    # fixating random seed for reproduction 
    seed = 1
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #model =  normalLayered(activation=["tanh"],penalization = [0.0],size=[400],dropout=[0.1],input_dropout=0.5)
    model =  modFunc()
    cv_x_train,cv_x_test,cv_y_train,cv_y_test = kfold_strat(train.drop("Class",axis=1),train["Class"],n_splits=n_folds,shuffle=False)
    testTrainModel(x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset,train["Class"].values,"main",model,compilationFunc,n_epochs,batch_size,statsFunc = statsFunc,saveFreq=saveFreq,deviceid=deviceid)
    for k in range(n_folds):
        # fixating random seed for reproduction 
        model = modFunc()
        testTrainModel(cv_x_train[k].values,cv_y_train[k].values,cv_x_test[k].values,cv_y_test[k].values,cv_y_train[k].values,"cv{0}".format(k+1),model,compilationFunc,n_epochs,batch_size,statsFunc = statsFunc,saveFreq=saveFreq,deviceid=deviceid)

        
def trainModel(x_train_data,y_train_data,x_test_data,y_test_data,y_true,modName,modFunc,compilationFunc,n_epochs,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    model = modFunc() 
    compilationFunc(model)
    print(saveFreq)
    result = pd.DataFrame() 
    initTime = time.perf_counter()
    for i in range(n_epochs):
        with tf.device(deviceid):
            print("{1}, epoch {0}".format(i+1,modName))
            model.fit(x_train_data,y_train_data,
                        batch_size = batch_size,
                      epochs=1,
                      validation_data=(x_test_data,y_test_data))                       
            stats = statsFunc(x_train_data,y_true,model)
            result = result.append(stats,ignore_index=True)
        if i % saveFreq == 0:
            if not(os.path.isdir(modFunc.__name__)):
                os.mkdir(modFunc.__name__)
            # saving main model
            saveModel(model,os.path.join(modFunc.__name__,modName))
            result.to_csv(os.path.join(modFunc.__name__,modName,"trainingStats.csv"),index=False)
            # saving predictions made from validation data
            y_valid = getPredictions(validation_dataset,model)
            np.savetxt(os.path.join(modFunc.__name__,modName,"predictions.csv"),y_valid,delimiter=",",fmt="%i" )
            # saving MetaParams
            pd.DataFrame.from_dict({"time":[time.perf_counter()-initTime],"batch size":[batch_size]}).to_csv(os.path.join(modFunc.__name__,"metaParams.csv"))
    del model, result

def crossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    # fixating random seed for reproduction 
    cv_x_train,cv_x_test,cv_y_train,cv_y_test = kfold_strat(train.drop("Class",axis=1),train["Class"],n_splits=n_folds,shuffle=False)
    trainModel(x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset,train["Class"].values,"main",modFunc,compilationFunc,n_epochs,batch_size,statsFunc = statsFunc,saveFreq=saveFreq,deviceid=deviceid)
    for k in range(n_folds):
        # fixating random seed for reproduction 
        trainModel(cv_x_train[k].values,cv_y_train[k].values,cv_x_test[k].values,cv_y_test[k].values,cv_y_train[k].values,"cv{0}".format(k+1),modFunc,compilationFunc,n_epochs,batch_size,statsFunc = statsFunc,saveFreq=saveFreq,deviceid=deviceid)

def specialTrainModel(x_train_data,y_train_data,x_test_data,y_test_data,y_true,modName,model,compilationFunc,n_epochs,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    compilationFunc(model)
    result = pd.DataFrame() 
    initTime = time.perf_counter()
    new_x_train_data = x_train_data.reshape(-1,28,28,1)
    new_x_test_data = x_test_data.reshape(-1,28,28,1)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,  zoom_range = 0.10,  width_shift_range=0.1, height_shift_range=0.1)
    for i in range(n_epochs):
        with tf.device(deviceid):
            print("{1}, epoch {0}".format(i+1,modName))
            model.fit(datagen.flow(new_x_train_data,y_train_data, batch_size = batch_size),
                        steps_per_epoch=int(np.floor(x_train_data.shape[0]/batch_size)),
                      epochs=1,
                      validation_data=(new_x_test_data,y_test_data))                       
            stats = statsFunc(new_x_train_data,y_true,model)
            result = result.append(stats,ignore_index=True)
        if i % saveFreq == 0:
            if not(os.path.isdir(modFunc.__name__)):
                os.mkdir(modFunc.__name__)
            # saving main model
            saveModel(model,os.path.join(modFunc.__name__,modName))
            result.to_csv(os.path.join(modFunc.__name__,modName,"trainingStats.csv"),index=False)
            # saving predictions made from validation data
            y_valid = getPredictions(validation_dataset.reshape(-1,28,28,1),model)
            np.savetxt(os.path.join(modFunc.__name__,modName,"predictions.csv"),y_valid,delimiter=",",fmt="%i" )
            # saving MetaParams
            pd.DataFrame.from_dict({"time":[time.perf_counter()-initTime],"batch size":[batch_size]}).to_csv(os.path.join(modFunc.__name__,"metaParams.csv"))
    del model, result

def specialCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    seed = 1
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #model =  normalLayered(activation=["tanh"],penalization = [0.0],size=[400],dropout=[0.1],input_dropout=0.5)
    model =  modFunc()
    cv_x_train,cv_x_test,cv_y_train,cv_y_test = kfold_strat(train.drop("Class",axis=1),train["Class"],n_splits=n_folds,shuffle=False)
    specialTrainModel(x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset,train["Class"].values,"main",model,compilationFunc,n_epochs,batch_size,statsFunc = statsFunc,saveFreq=saveFreq,deviceid=deviceid)
    for k in range(n_folds):
        model =  modFunc()
        specialTrainModel(cv_x_train[k].values,cv_y_train[k].values,cv_x_test[k].values,cv_y_test[k].values,cv_y_train[k].values,"cv{0}".format(k+1),model,compilationFunc,n_epochs, batch_size,statsFunc = statsFunc, saveFreq=saveFreq, deviceid=deviceid)



compilationFunc = interestingCompile
n_epochs = 101
n_folds = 10
batch_size = 128
freq = 5

modFunc = lambda : normalLayered(activation=["tanh"],penalization = [0.0],size=[400],dropout=[0.1],input_dropout=0.5)
modFunc.__name__ = "normal1" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = lambda : twoLayers(activation=["tanh","swish"],penalization=[0.005,0.001],size=[200,100])
modFunc.__name__ = "normal2" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = lambda : twoLayersDropout(activation=["tanh","tanh"],penalization=[0.005,0.001],size=[200,100], dropout=0.5)
modFunc.__name__ = "normal3" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = lambda : normalLayered(activation=["tanh","tanh","tanh"],penalization=[0.001,0.0,0.0],size=[200,100,100], dropout=[0.0,0.0,0.0],input_dropout=0.2)
modFunc.__name__ = "normal4" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = lambda : normalLayered(activation=["tanh","relu","tanh"],penalization=[0.001,0.00,0.00],size=[200,150,100], dropout=[0.0,0.0,0.0],input_dropout=0.2)
modFunc.__name__ = "normal5" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = lambda : normalLayered(activation=["tanh","swish","tanh","swish",],penalization=[0.001,0.00,0.00,0.00],size=[200,150,100,100], dropout=[0.0,0.0,0.0,0.0] ,input_dropout=0.2)
modFunc.__name__ = "normal6" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

n_epochs = 51

modFunc = convMod5
modFunc.__name__ = "conv5" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
specialCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")


modFunc = convMod1
modFunc.__name__ = "conv1" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = convMod2
modFunc.__name__ = "conv2" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = convMod3
modFunc.__name__ = "conv3" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = convMod4
modFunc.__name__ = "conv4" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = convMod5
modFunc.__name__ = "conv5" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

modFunc = convMod6
modFunc.__name__ = "conv6" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

