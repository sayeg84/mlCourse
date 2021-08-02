
import argparse
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
import time
import tools

parser= argparse.ArgumentParser(description="Adjust a logistic regression using SKLearn")
parser.add_argument("--full_data",help="Whether to train on full data or on smaller batch", type=eval,choices=[False,True],default=True)
parser.add_argument("--n_folds", type=int,default=5)
parser.add_argument("--solver", type=str,default="saga")
args=parser.parse_args()

folderName = "logReg"
if not(os.path.isdir(folderName)):
    os.mkdir(folderName)

print("Reading data")
print()

test = pd.read_csv("MNISTtest.csv")
train = pd.read_csv("MNISTtrain.csv")
validation = pd.read_csv("MNISTvalidate.csv")
names = {"C785":"Class"}
test = test.rename(columns=names)
train = train.rename(columns=names)
validation = validation.rename(columns=names)
#standarizing
test.iloc[:,:-1] = test.iloc[:,:-1].multiply(1/255)
train.iloc[:,:-1] = train.iloc[:,:-1].multiply(1/255)
validation.iloc[:,:-1] = validation.iloc[:,:-1].multiply(1/255)

print("Done")
print()

if not args.full_data:
    print("Training on test data")
    print()
    data = test
else:
    print("Training on full data")
    print()
    data = train

print("Making clasification")
print()

initTime = time.perf_counter()
logReg = LogisticRegression(penalty="none",max_iter=10**7,solver=args.solver,verbose=True)
logReg.fit(data.drop("Class",axis=1),data["Class"])

print("Done")
print()
y_pred = logReg.predict(test.drop("Class",axis=1))
if not(os.path.isdir(os.path.join(folderName,"main"))):
    os.mkdir(os.path.join(folderName,"main"))
np.savetxt(os.path.join(folderName,"main","y_pred_log_reg.csv"),y_pred,delimiter=",")
np.savetxt(os.path.join(folderName,"main","y_pred_validate_log_reg.csv"),logReg.predict(validation),delimiter=",")
np.savetxt(os.path.join(folderName,"main","time.csv"),np.array([time.perf_counter()-initTime]),delimiter=",")
np.savetxt(os.path.join(folderName,"main","predictions.csv"),np.array(classStatistics(y_pred,test["Class"].values)),delimiter=",")
pickle.dump(logReg,open(os.path.join(folderName,"main","model"),"wb"))
err = (y_pred - test["Class"]) != 0
print(1-np.mean(err))
print("Time spent: ")
print(np.round(time.perf_counter()-initTime,2),end="")
print(" seconds")
X_train,X_test,y_train,y_test = kfold_strat(data.drop("Class",axis=1),data["Class"],n_splits=args.n_folds,shuffle=True)
for i in range(args.n_folds):
    print("Making CV iter {0}".format(i+1))
    print()
    #classes,classifiers = oneVsRestLogisticRegrresionClassifier(test.drop("Class",axis=1),test["Class"],verbo=True,maxiter=10**4)
    initTime = time.perf_counter()
    logReg = LogisticRegression(penalty="none",max_iter=10**7,solver=args.solver,verbose=True)
    logReg.fit(X_train[i],y_train[i])
    print("Done")
    print()
    y_pred = logReg.predict(X_test[i])
    zerosN = int(np.ceil(np.log10(args.n_folds)))+1
    folder = "sub{0}".format(str(i+1).zfill(zerosN))
    if not(os.path.isdir(os.path.join(folderName,folder))):
        os.mkdir(os.path.join(folderName,folder))
    np.savetxt(os.path.join(folderName,folder,"y_pred_log_reg.csv"),y_pred,delimiter=",")
    np.savetxt(os.path.join(folderName,folder,"time.csv"),np.array([time.perf_counter()-initTime]),delimiter=",")
    np.savetxt(os.path.join(folderName,folder,"predictions.csv"),np.array(classStatistics(y_pred,y_test[i].values)),delimiter=",")
    pickle.dump(logReg,open(os.path.join(folderName,folder,"model"),"wb"))
    err = (y_pred - y_test[i]) != 0
    print(1-np.mean(err))
    print("Time spent: ")
    print(np.round(time.perf_counter()-initTime,2),end="")
    print(" seconds")
# save labelsO si p
labels = labelData(train,y_test)
labels.to_csv(os.path.join(modFunc.__name__,"test_classes.csv"),index=True)
