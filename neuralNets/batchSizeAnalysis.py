#!/usr/bin/env python
# coding: utf-8

# In[7]:

from datetime import datetime
import gc
import models
import numpy as np
import os
import pandas as pd
import random
import run
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




def batchSizeAnalysis(modFunc,compilationFunc,n_epochs,n_folds,batch_size,statsFunc = basicStatistics,saveFreq=5,deviceid="GPU:0"):
    # fixating random seed for reproduction 
    seed = 1
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #model =  normalLayered(activation=["tanh"],penalization = [0.0],size=[400],dropout=[0.1],input_dropout=0.5)
    model =  modFunc()
    #cv_x_train,cv_x_test,cv_y_train,cv_y_test = kfold_strat(train.drop("Class",axis=1),train["Class"],n_splits=n_folds,shuffle=False)
    mod, res = testTrainModel(x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset,train["Class"].values,"main",model,compilationFunc,n_epochs,batch_size,statsFunc = statsFunc,saveFreq=saveFreq,deviceid=deviceid)
    return mod, res
        


# In[ ]:


compilationFunc = interestingCompile
n_epochs = 21
n_folds=1
freq = 52
models = []
results = []
for batch_size in [1,2,4,8,16,32,64,128,256]:
    modFunc = oneLayer1
    modFunc.__name__ = "bs{0}".format(batch_size) + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    mod, res = batchSizeAnalysis(modFunc,compilationFunc,n_epochs,n_folds,batch_size,saveFreq = freq,statsFunc = classStatistics,deviceid="GPU:0")
    models.append(mod)
    results.append(res)


# In[41]:


plt.figure(figsize=(4.5,3))
plt.plot([1,2,4,8,16,32,64,128,256],[1-res["val_accuracy"].iloc[-1] for res in results],marker="o",color="C3")
#plt.plot([1,2,4,8,16,32,64,128,256],[res["val_loss"].iloc[-1] for res in results],marker="o")
#plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.xlabel("Batch size",fontsize=14)
plt.ylabel("val error",fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join("batchSizeAnalError.pdf"))
plt.show()
plt.figure(figsize=(4.5,3))
#plt.plot([1,2,4,8,16,32,64,128,256],[1-res["val_accuracy"].iloc[-1] for res in results],marker="o",color="C2")
plt.plot([1,2,4,8,16,32,64,128,256],[res["val_loss"].iloc[-1] for res in results],marker="o",color="C2")
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.xlabel("Batch size",fontsize=14)
plt.ylabel("val loss",fontsize=14)
plt.tight_layout()
plt.savefig("batchSizeAnalLoss.pdf")
plt.show()


# In[39]:





# In[ ]:




