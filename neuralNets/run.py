#!/usr/bin/env python
# coding: utf-8


from datetime import datetime
import models
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import time
import tools




# fixating random seed for reproduction 

seed = 1
os.environ['PYTHONHASHSEED']=str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



print()                                                            
print("reading Data")

test = pd.read_csv("MNISTtest.csv")
train = pd.read_csv("MNISTtrain.csv")
validation = pd.read_csv("MNISTvalidate.csv")
names = {"C785":"Class"}
test = test.rename(columns=names)
train = train.rename(columns=names)
validation = validation.rename(columns=names)
#standarizing
#test.iloc[:,:-1] = test.iloc[:,:-1].multiply(1/255)
#train.iloc[:,:-1] = train.iloc[:,:-1].multiply(1/255)
#validation.iloc[:,:-1] = validation.iloc[:,:-1].multiply(1/255)


x_test_dataset = dataScaler(test.drop("Class",axis=1)).values
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


if __name__ == "__main__":

    compilationFunc = interestingCompile
    n_epochs = 101
    n_folds = 10
    batch_size = 128
    freq = 5

    modFunc = lambda : normalLayered(activation=["tanh"],penalization = [0.0],size=[400],dropout=[0.1],input_dropout=0.5)
    modFunc.__name__ = "DNN1" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = lambda : twoLayers(activation=["tanh","swish"],penalization=[0.005,0.001],size=[200,100])
    modFunc.__name__ = "DNN2" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = lambda : twoLayersDropout(activation=["tanh","tanh"],penalization=[0.005,0.001],size=[200,100], dropout=0.5)
    modFunc.__name__ = "DNN3" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = lambda : normalLayered(activation=["tanh","tanh","tanh"],penalization=[0.001,0.0,0.0],size=[200,100,100], dropout=[0.0,0.0,0.0],input_dropout=0.2)
    modFunc.__name__ = "DNN4" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = lambda : normalLayered(activation=["tanh","relu","tanh"],penalization=[0.001,0.00,0.00],size=[200,150,100], dropout=[0.0,0.0,0.0],input_dropout=0.2)
    modFunc.__name__ = "DNN5" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = lambda : normalLayered(activation=["tanh","swish","tanh","swish",],penalization=[0.001,0.00,0.00,0.00],size=[200,150,100,100], dropout=[0.0,0.0,0.0,0.0] ,input_dropout=0.2)
    modFunc.__name__ = "DNN6" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    n_epochs = 51

    modFunc = convMod6
    modFunc.__name__ = "CNN6" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    specialCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")


    modFunc = convMod1
    modFunc.__name__ = "CNN1" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = convMod2
    modFunc.__name__ = "CNN2" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = convMod3
    modFunc.__name__ = "CNN3" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = convMod4
    modFunc.__name__ = "CNN4" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

    modFunc = convMod5
    modFunc.__name__ = "CNN5" + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    testCrossValidationTraining(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")

