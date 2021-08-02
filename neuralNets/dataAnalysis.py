#!/usr/bin/env python
# coding: utf-8

# In[268]:



# %%
import argparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
# custom library
import tools

parser= argparse.ArgumentParser(description="Make plots for a model")
parser.add_argument("--path", type=str,required=True)
parser.add_argument("--format", type=str,choices=["png","pdf"],default="pdf")
parser.add_argument("--type", type=str,choices=["full","simple","folderFull","folderSimple"],default="simple")
parser.add_argument("--save", type=str,choices=["true","false"],default="false")
args=parser.parse_args()



def plotDigit(dataFrame,obs,predicted=[],**kwargs):
    if isinstance(obs,int):
        zs = dataFrame.iloc[obs,:-1]
        zs = np.reshape(zs.to_numpy(),(28,28))
        plt.imshow(zs,**kwargs)
        if len(predicted)!=0:
            plt.title("Real digit: {0} \n Predicted: {1}".format(dataFrame.loc[obs,"Class"],predicted[0]))
        else:
            plt.set_title("Real digit: {0}".format(dataFrame.loc[obs,"Class"]))
    elif isinstance(obs,list):
        n = int(np.ceil(np.sqrt(len(obs))))
        fig,axs = plt.subplots(nrows=n,ncols=n,sharex=True,sharey=True)
        extended = np.ravel(axs)
        for index in range(n*n):
            if index < len(obs):
                dig = obs[index]
                zs = dataFrame.iloc[dig,:-1]
                zs = np.reshape(zs.to_numpy(),(28,28))
                extended[index].imshow(zs,**kwargs)
                if len(predicted)!=0:
                    extended[index].set_title("Real digit: {0} \n Predicted: {1}".format(dataFrame.loc[dig,"Class"],predicted[index]))
                else:
                    extended[index].set_title("Real digit: {0}".format(dataFrame.loc[dig,"Class"]))
            else:
                fig.delaxes(extended[index])

def plotPredictionProbs(dataFrame,model,index):
    pred = getSinglePredictions(dataScaler(dataFrame.drop("Class",axis=1)).values,model,index)
    probs = getPredictionProbs(dataScaler(dataFrame.drop("Class",axis=1)).values,model,index)
    pred = np.argmax(probs)
    axs = plt.subplot(1,2,1)
    plotDigit(dataFrame,index,[pred],cmap=cm.Greys)
    axs = plt.subplot(1,2,2,aspect="auto")
    plt.bar(range(10),probs)
    plt.yscale("log")
    plt.xticks(range(10),labels=[str(i) for i in range(10)])
    plt.title("Probabilities")

def plotAccuracy(result,error=False):
    if error:
        ys1 = [1-y for y in result['accuracyRate']]
        ys2 = [1-y for y in result['val_accuracy']]
        l1 = "Error rate"
    else:
        ys1 = result['accuracyRate'].values
        ys2 = result['val_accuracy'].values
        l1 = "Error Accuracy"
    plt.plot(ys1,label = "Training")
    plt.plot(ys2,label = "Validation")
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel(l1)
    plt.legend(loc = "upper left")

def plotClassesAccuracy(result,error=False):
    classesCols = [name for name in result.columns if "Class" in  name]
    for stat in classesCols:
        if error:
            plt.plot([1-y for y in result[stat]],label = "Class {0}".format(stat[-1]))
            plt.ylabel("Error rate")
        else:
            plt.plot(result[stat],label = "Class {0}".format(stat[-1]))
            plt.ylabel("Accuracy")
    plt.grid()
    plt.xlabel("Epochs")
    plt.legend(bbox_to_anchor=(1,1))

def plotCrossValidation(results,error=False):
    fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(10,4))
    if error:
        axs[0].plot([1-y for y in results[0]["accuracyRate"]],label = "Full")
        r = []
        for i,res in enumerate(results[1:]):
            axs[0].plot([1-y for y in res["accuracyRate"]],label = "CV {0}".format(i))
            r.append(1-res["accuracyRate"].iloc[-1])
        axs[0].set_ylabel("Error rate")
    else:
        axs[0].plot(results[0]["accuracyRate"],label = "Full")
        r = []
        for i,res in enumerate(results[1:]):
            axs[0].plot(res["accuracyRate"],label = "CV {0}".format(i))
            r.append(res["accuracyRate"].iloc[-1])
        axs[0].set_ylabel("Accuracy rate")
    axs[0].plot([0,np.max(results[0].index)],[np.mean(r),np.mean(r)],linestyle="--",color="k")
    axs[0].grid()
    axs[0].set_xlabel("epochs")
    axs[0].set_title("Training")
    if error:
        axs[1].plot([1-y for y in results[0]["val_accuracy"]],label = "Full")
        r = []
        for i,res in enumerate(results[1:]):
            axs[1].plot([1-y for y in res["val_accuracy"]],label = "CV {0}".format(i))
            r.append(1-res["val_accuracy"].iloc[-1])
    else:
        axs[1].plot(results[0]["val_accuracy"],label = "Full")
        r = []
        for i,res in enumerate(results[1:]):
            axs[1].plot(res["val_accuracy"],label = "CV {0}".format(i))
            r.append(res["val_accuracy"].iloc[-1])
    axs[1].plot([0,np.max(results[0].index)],[np.mean(r),np.mean(r)],linestyle="--",color="k")
    axs[1].grid()
    axs[1].set_xlabel("Epochs")
    axs[1].set_title("Validation")
    axs[1].legend(bbox_to_anchor=(1,1))


def plotCrossValCompar(results,error=False):
    fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(10,4))
    if error:
        axs[0].plot([1-y for y in results[0]["accuracyRate"]],label = "Training")
        axs[0].plot([1-y for y in results[0]["val_accuracy"]],label = "Validation")
        axs[0].set_ylabel("Error rate")
    else:
        axs[0].plot(results[0]["accuracyRate"],label = "Training")
        axs[0].plot(results[0]["val_accuracy"],label = "Validation")
        axs[0].set_ylabel("Accuracy rate")
        #axs[0].plot([0,np.max(results[0].index)],[np.mean(r),np.mean(r)],linestyle="--",color="k")
    axs[0].grid()
    axs[0].set_xlabel("Epochs")
    axs[0].set_title("Training")
    r1 = []
    r2 = []
    if error:
        for i,res in enumerate(results[1:]):
            r1.append([1-y for y in res["accuracyRate"].values])
            r2.append([1-y for y in res["val_accuracy"].values])
    else:
        for i,res in enumerate(results[1:]):
            r1.append(res["accuracyRate"].values)
            r2.append(res["val_accuracy"].values)
    axs[1].errorbar(range(len(r1[0])),np.mean(r1,axis=0),yerr=np.std(r1,axis=0),alpha=0.8,capsize=3,fmt="o",label="Training")
    axs[1].errorbar(range(len(r1[0])),np.mean(r2,axis=0),yerr=np.std(r2,axis=0),alpha=0.8,capsize=3,fmt="o",label="Validation")
    #axs[1].plot([0,np.max(results[0].index)],[np.mean(r),np.mean(r)],linestyle="--",color="k")
    axs[1].grid()
    axs[1].set_xlabel("Epochs")
    axs[1].set_title("Cross validation comparison")
    axs[1].legend(bbox_to_anchor=(1,1))

def plotCrossValGrid(results,error=False):
    fig,axs = plt.subplots(nrows=4,ncols=3,figsize=(12,8),sharex=True,sharey=True)
    if error:
        l1 = "Error rate"
        axs[0][0].plot([1-y for y in results[0]["accuracyRate"].values],label="Training",lw=4)
        axs[0][0].plot([1-y for y in results[0]["val_accuracy"].values],label="Validation",lw=4)
        for i,ax in enumerate(axs.flatten()[1:]):
            if i+1<len(results):
                ax.plot([1-y for y in results[i+1]["accuracyRate"].values],label="Training",lw=2)
                ax.plot([1-y for y in results[i+1]["val_accuracy"].values],label="Validation",lw=2)
                ax.grid()
            else:
                fig.delaxes(ax)
    else:
        l1 = "Accuracy rate"
        axs[0][0].plot(results[0]["accuracyRate"].values,label="Training",lw=4)
        axs[0][0].plot(results[0]["val_accuracy"].values,label="Validation",lw=4)
        for i,ax in enumerate(axs.flatten()[1:]):
            if i+1<len(results):
                ax.plot(results[i+1]["accuracyRate"].values,label="Training",lw=2)
                ax.plot(results[i+1]["val_accuracy"].values,label="Validation",lw=2)
                ax.grid()
            else:
                fig.delaxes(ax)
    axs[0][0].grid()
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.90,0.3),fontsize=20)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    fig.text(0.5,0.03,"Epochs",fontsize=24)
    fig.text(0.06,0.5,l1,ha='center',va='center',fontsize=24,rotation=90)

def plotConfusionMatrix(y_true,y_pred,cmap = cm.Blues,text=False,fs=10,thresh=0.01,rot=45):
    mat = confusion_matrix(y_true,y_pred,normalize="true")
    norm = mcolors.SymLogNorm(1e-5,vmin=0,vmax=1)
    if text:
        plt.imshow(mat,norm=norm,cmap=cmap)
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            plt.text(j,i, "{0:.4f}".format(mat[i, j]),
                     ha="center",va="center",
                     color="white" if mat[i, j] > thresh else "black",
                     fontsize=fs,rotation=rot)
    else:
        plt.imshow(mat,norm=norm,cmap=cmap)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_label_position("right")
    #cbar.ax.tick_params(labelsize=1.5*f)
    cbar.ax.set_ylabel("Proportion")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    classes = [str(i) for i in range(len(set(y_true)))]
    plt.xticks(range(len(set(y_true))),classes)
    plt.yticks(range(len(set(y_true))),classes)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

def plotModelsAccuracyGrid(resultsArray,names,error=False):
    fig,axs = plt.subplots(nrows=4,ncols=3,figsize=(12,8),sharex=True,sharey=True)
    if error:
        l1 = "Error rate"
        for i,ax in enumerate(axs.flatten()):
            ax.plot([1-y for y in resultsArray[i][0]["accuracyRate"].values],label="Training",lw=2)
            ax.plot([1-y for y in resultsArray[i][0]["val_accuracy"].values],label="Validation",lw=2)
            ax.set_title(names[i])
            ax.grid()
    else:
        l1 = "Accuracy rate"
        for i,ax in enumerate(axs.flatten()[1:]):
            ax.plot(resultsArray[i][0]["accuracyRate"].values,label="Training",lw=2)
            ax.plot(resultsArray[i][0]["val_accuracy"].values,label="Validation",lw=2)
            ax.set_title(names[i])
            ax.grid()
    handles, labels = axs[0][0].get_legend_handles_labels()
    plt.subplots_adjust(wspace=0.05)
    fig.legend(handles, labels, bbox_to_anchor=(0.9,0.90),fontsize=20)
    fig.text(0.5,0.03,"Epochs",fontsize=24)
    fig.text(0.05,0.5,l1,ha='center',va='center',fontsize=24,rotation=90)

def confusionMatrixTable(y_true,y_pred):
    mat = confusion_matrix(y_true,y_pred,normalize='true')
    mat = pd.DataFrame(mat)
    mat.index.name = "Actual"
    mat.columns.name = "Predicted"
    return mat


def batchAnalysis(direc="models"):
    folders = [name for name in os.listdir(direc) if os.path.isdir(os.path.join(direc,name)) and not("." in name)]
    batchFolders = [name for name in folders if name[0:5] == "batch"]
    batchFolders = sorted(batchFolders)
    batchValues = [int(name[-5:]) for name in batchFolders]
    batchModels = [readCrossValidationModel(os.path.join(direc,name)) for name in batchFolders]
    ys = [batch[1][0].iloc[-1]["val_accuracy"] for batch in batchModels]
    bestValAccuracies = [batchValues[i] for i in np.argsort(ys)]
    bestValAccuracies.reverse()
    plt.scatter(batchValues,ys,label="val_accuracy")
    plt.plot(batchValues,ys)
    ys = [batch[1][0].iloc[-1]["accuracyRate"] for batch in batchModels]
    bestAccuracies = [batchValues[i] for i in np.argsort(ys)]
    bestAccuracies.reverse()
    plt.scatter(batchValues,ys,label="accuracyRate")
    plt.plot(batchValues,ys)
    plt.xscale("log")
    plt.grid()

def getNormalTable(results,columns=["loss","accuracyRate","val_loss","val_accuracy"]):
    xval = pd.DataFrame(columns=columns)
    new = pd.concat([res.loc[[res.shape[0]-1],columns] for res in results[1:]],ignore_index=True)
    return new

def getCrossValTable(results,columns=["loss","accuracyRate","val_loss","val_accuracy"]):
    xval = pd.DataFrame(columns=columns)
    new = pd.concat([res.loc[[res.shape[0]-1],columns] for res in results[1:]],ignore_index=True)
    return new

def getSummaryStatistics(results,columns=["loss","accuracyRate","val_loss","val_accuracy"],error=False):
    new = getCrossValTable(results,columns)
    res = results[0].loc[[results[0].shape[0]-1],columns].copy()
    res.index = [0]
    if error:
        for var in res.columns:
            if "accuracy" in var:
                res[var] = 1-res[var]
        res.columns = [s.replace("accuracy","error") for s in res.columns]
        for var in new.columns:
            if "accuracy" in var:
                new[var] = 1-new[var]
        new.columns = [s.replace("accuracy","error") for s in new.columns]
    meandf = pd.DataFrame(new.mean()).transpose()
    meandf.columns = ["cv_" + st +"_mean" for st in meandf.columns]
    stddf = pd.DataFrame(new.std()).transpose()
    stddf.columns = ["cv_" + st + "_std" for st in stddf.columns]
    return pd.concat([res,meandf,stddf],axis=1)


def modelCVComparisonTable(resultsArray,columns = ["loss","accuracyRate","val_loss","val_accuracy"],error=False):
    return pd.concat([getSummaryStatistics(results,columns,error=error) for results in resultsArray])

def modelComparisonTable(resultsArray,columns = ["loss","accuracyRate","val_loss","val_accuracy"],error=False):
    res = pd.concat([results[0].loc[[results[0].shape[0]-1],columns] for results in resultsArray])
    if error:
        for var in res.columns:
            if "accuracy" in var:
                res[var] = 1-res[var]
        res.columns = [s.replace("accuracy","error") for s in res.columns]
    return res

def trainableParams(model):
    total = 0
    for var in model.trainable_variables:
        total += var.numpy().size
    return total

def getLayers(model):
    return len(models[0].layers)-2

def getDenseLayers(model):
    # beggining at -1 to remove last layer
    c=-1
    for lay in model.layers:
        if type(lay) == tf.keras.layers.Dense:
            c += 1
    return c

def getConvLayers(model):
    c=0
    for lay in model.layers:
        if type(lay) == tf.keras.layers.Conv2D:
            c += 1
    return c

def modelParametersTable(modelsArray):
    res = pd.DataFrame({"Trainable_Parameters":[trainableParams(models[0]) for models in modelsArray],
                        "Dense_Layers":[getDenseLayers(models[0]) for models in modelsArray], 
                        "Convolution_Layers":[getConvLayers(models[0]) for models in modelsArray]})
    return res

def executionTime(path):
    return 11*pd.read_csv(os.path.join(path,"metaParams.csv"),index_col=0,header=0).iloc[0,0]/3600

def plotModelComparisonTable(table):
    for col in table.columns:
        if "time" not in col:
            plt.plot(table[col],label=col,marker="o")
    plt.xticks(range(table.shape[0]),labels=table.index.to_list(),rotation=45)
    plt.xlabel("Model")
    plt.ylabel("Val")
    plt.grid()
    plt.legend(loc="upper left")


def perClassStatisticsTable(models,results,error=False):
    testMat = []
    trainMat = []
    x_train,x_test,y_train,y_test = kfold_strat(train.drop("Class",axis=1),train["Class"])
    print("Analyzing main")
    y_pred_test = getPredictions(dataScaler(test.drop("Class",axis=1)).values,models[0])
    y_pred_train = getPredictions(dataScaler(train.drop("Class",axis=1)).values,models[0])
    testMat.append(confusionMatrixTable(test["Class"].values,y_pred_test))
    trainMat.append(confusionMatrixTable(train["Class"].values,y_pred_train))
    for index,model in enumerate(models[1:]):
        print("Analyzing cv {0}".format(index))
        y_pred_test = getPredictions(dataScaler(x_test[index]).values,model)
        y_pred_train = getPredictions(dataScaler(x_train[index]).values,model)
        testMat.append(confusionMatrixTable(y_text[index].values,y_pred_test))
        trainMat.append(confusionMatrixTable(y_train[index].values,y_pred_train))
    n_class = set(train["Class"].values)
    res1 = pd.DataFrame(columns=range(10))
    res1["error"] = [1 - traintMat[0][i][i] for i in range(10)]
    res1["val error"] = [1 - testMat[0][i][i] for i in range(10)]
    aux  = [[1 - testMat[j][i][i] for i in range(10)] for j in range(1,len(models)+1)]
    res1["cv val error mean"] = np.mean(aux,axis=0)
    res1["cv val error std"] = np.std(aux,axis=0)
    return res1
    
def makeFullPlots(path,error=False,save=False,f = "pdf"):
    
    print("Final validation error rate: ")
    print(1-results[0].loc[results[0].shape[0]-1,"val_accuracy"])
    print(models[0].summary())

    fig = plt.figure()
    plotAccuracy(results[0],error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"accuracy.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    fig = plt.figure(figsize=(7,4))
    
    plotClassesAccuracy(results[0],error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"classAccuracy.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    
    plotCrossValidation(results,error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"crossValidation.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    
    plotCrossValCompar(results,error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"crossValCompar.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    
    plotCrossValGrid(results,error=error)
    #plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"crossValGrid.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    mistakes = getWrongPredictions(dataScaler(test.drop("Class",axis=1)),models[0],test["Class"].values)
    fig = plt.figure(figsize=(8,4))
    plotPredictionProbs(test,models[0],int(mistakes[0]))
    if save:
        plt.savefig(os.path.join(path,"predictionExample.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    plt.show()
    plotConfusionMatrix(train["Class"].values,getPredictions(dataScaler(train.drop("Class",axis=1)).values,models[0]),text=True,fs=6,rot=35)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"confussionMatTrain.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    plotConfusionMatrix(test["Class"].values,getPredictions(dataScaler(test.drop("Class",axis=1)).values,models[0]),text=True,fs=6,rot=35)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"confussionMatTest.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    plt.show()
    table = confusionMatrixTable(test["Class"].values,getPredictions(dataScaler(test.drop("Class",axis=1)).values,models[0]))
    table.to_latex(buf=os.path.join(path,"main","testConfusion.csv"),float_format="{:4f}".format)
    table = confusionMatrixTable(train["Class"].values,getPredictions(dataScaler(train.drop("Class",axis=1)).values,models[0]))
    table.to_latex(buf=os.path.join(path,"main","trainConfusion.csv"),float_format="{:4f}".format)




def makeSimplePlots(path,error=False,save=False,f = "pdf"):
    results = readCrossValidationModel(path,model=False)
    print("Final validation error rate: ")
    print(1-results[0].loc[results[0].shape[0]-1,"val_accuracy"])
    fig = plt.figure()
    plotAccuracy(results[0],error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"accuracy.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    fig = plt.figure(figsize=(7,4))
    
    plotClassesAccuracy(results[0],error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"classAccuracy.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    
    plotCrossValidation(results,error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"crossValidation.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    
    plotCrossValCompar(results,error=error)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"crossValCompar.{0}".format(f)))
        plt.close()
    else:
        plt.show()
    
    plotCrossValGrid(results,error=error)
    #plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path,"crossValGrid.{0}".format(f)))
        plt.close()
    else:
        plt.show()



if args.type =="full":
    print()
    print("Reading data")
    print()
    test = pd.read_csv("MNISTtest.csv")
    train = pd.read_csv("MNISTtrain.csv")
    validation = pd.read_csv("MNISTvalidate.csv")
    names = {"C785":"Class"}
    test = test.rename(columns=names)
    train = train.rename(columns=names)

    
    print()
    print("Importing Tensor Flow")
    print()
    import tensorflow as tf
    
    models,results = readCrossValidationModel(args.path,model=True)
    #makeFullPlots(args.path,error=True,save=args.save,f = args.format)
    table = perClassStatisticsTable(results,models)
    print(table)
    table.to_latex(buf=os.path.join(args.path,"classError.tex"),float_format="{}")

elif args.type =="simple":    
    makeSimplePlots(args.path,error=True,save=args.save,f = args.format)

elif args.type == "folderSimple":
    folder = args.path
    resultsArray = []
    names = []
    times = []
    for directory in os.listdir(folder):
        if os.path.isdir(os.path.join(folder,directory)) and directory[0]!=".":
            print("reading folder {0}".format(directory))
            results = readCrossValidationModel(os.path.join(folder,directory),model=False)
            # models[0].save(os.path.join(folder,directory,"model.h5"))
            resultsArray.append(results)
            names.append(directory.split("_")[0])
            times.append(executionTime(os.path.join(folder,directory)))
    plotModelsAccuracyGrid(resultsArray,names,error=True)
    #plt.tight_layout()
    plt.savefig(os.path.join("reporte","modelsGrid.{0}".format(args.format)))
    #plt.show()
    
    table = modelComparisonTable(resultsArray,error=True)
    table = table.rename(columns={"errorRate": "error"})
    table["execution _time_(hours)"] = times
    table.index = names
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    plt.figure(figsize=(6,3))
    plotModelComparisonTable(table)
    plt.tight_layout()
    plt.savefig(os.path.join("reporte","modelsPlot.{0}".format(args.format)))
    
    for var in table.columns:
        if ("error" in var) or ("accuracy" in var):
            table[var] = 100*table[var]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","modelComparisonTable.tex"),float_format="{:0.4f}".format,column_format=tab)
    """
    table = modelCVComparisonTable(resultsArray,columns=["val_accuracy","val_loss"],error=True)
    table.index = names
    table = table.rename(columns={"errorRate": "error"})
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    for var in table.columns:
        if ("error" in var) or ("accuracy" in var):
            table[var] = 100*table[var]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","modelCVComparisonTable.tex"),float_format="{:0.4f}".format,column_format=tab)
    """
    table1 = modelCVComparisonTable(resultsArray,columns=["val_accuracy"],error=True)
    table2 = modelCVComparisonTable(resultsArray,columns=["val_loss"],error=True)
    table3 = modelComparisonTable(resultsArray,columns=["accuracy","loss"],error=True)
    table1.index = names
    table2.index = names
    table3.index = names
    table = pd.concat([table3[["error"]],table1,table3[["loss"]],table2],axis=1)
    table = table.rename(columns={"errorRate": "error"})
    print(table.columns)
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    for var in table.columns:
        if ("error" in var) or ("accuracy" in var):
            table[var] = 100*table[var]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","mainTable.tex"),float_format="{:1.3E}".format,column_format=tab)
    table.to_latex(buf=os.path.join("reporte","mainTableFloat.tex"),float_format="{:0.4f}".format,column_format=tab)

elif args.type == "folderFull":
    print()
    print("importing tensorflow")
    print()
    import tensorflow as tf
    folder = args.path
    modelsArray = []
    resultsArray = []
    names = []
    times = []
    for directory in os.listdir(folder):
        if os.path.isdir(os.path.join(folder,directory)) and directory[0]!=".":
            print("reading folder {0}".format(directory))
            models,results = readCrossValidationModel(os.path.join(folder,directory),model=True)
            # models[0].save(os.path.join(folder,directory,"model.h5"))
            resultsArray.append(results)
            modelsArray.append(models)
            names.append(directory.split("_")[0])
            times.append(executionTime(os.path.join(folder,directory)))
    plotModelsAccuracyGrid(resultsArray,names,error=True)
    plt.tight_layout()
    #   plt.show()
    plt.savefig(os.path.join("reporte","modelsGrid.{0}".format(args.format)))   

    
    table = modelComparisonTable(resultsArray,error=True)
    table = table.rename(columns={"errorRate": "error"})
    table.index = names
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","modelComparisonTable.tex"),float_format="{:0.4f}".format,column_format=tab)

    plt.figure(figsize=(6,3))
    plotModelComparisonTable(table)
    plt.tight_layout()
    plt.savefig(os.path.join("reporte","modelsPlot.{0}".format(args.format)))
    """
    table = modelCVComparisonTable(resultsArray,columns=["val_accuracy","val_loss"],error=True)
    table.index = names
    table = table.rename(columns={"errorRate": "error"})
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    for var in table.columns:
        if ("error" in var) or ("accuracy" in var):
            table[var] = 100*table[var]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","modelCVComparisonTable.tex"),float_format="{:0.4f}".format,column_format=tab)
    """
    table = modelParametersTable(modelsArray)
    table.index = names
    table["execution _time_(hours)"] = times
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    for var in table.columns:
        if ("error" in var) or ("accuracy" in var):
            table[var] = 100*table[var]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","modelParametersTable.tex"),float_format="{:0.4f}".format,column_format=tab)


    table1 = modelCVComparisonTable(resultsArray,columns=["val_accuracy"],error=True)
    table2 = modelCVComparisonTable(resultsArray,columns=["val_loss"],error=True)
    table3 = modelComparisonTable(resultsArray,columns=["accuracy","loss"],error=True)
    table = pd.concat([table3[["error"]],table1,table3[["loss"]],table2],axis=1)
    table.index = names
    table = table.rename(columns={"errorRate": "error"})
    table.columns = [" ".join(name.split("_")) for name in table.columns]
    for var in table.columns:
        if ("error" in var) or ("accuracy" in var):
            table[var] = 100*table[var]
    tab = "c"
    for i in table.columns:
        tab += "| p{1.5cm}"
    print(table)
    table.to_latex(buf=os.path.join("reporte","mainTable.tex"),float_format="{:0.4f}".format,column_format=tab)

elif args.type == "batchSize":
    resultsArray = []
    names = []
    times = []
    for directory in os.listdir(folder):
        if os.path.isdir(os.path.join(folder,directory)) and directory[0]!=".":
            print("reading folder {0}".format(directory))
            results = readCrossValidationModel(os.path.join(folder,directory),model=False)
            # models[0].save(os.path.join(folder,directory,"model.h5"))
            resultsArray.append(results)
            names.append(directory.split("_")[0])
            times.append(executionTime(os.path.join(folder,directory)))
    plt.figure(figsize=(4.5,3))
    plt.plot([1,2,4,8,16,32,64,128,256],[1-res["val_accuracy"].iloc[-1] for res in resultsArray],marker="o",color="C3")
    #plt.plot([1,2,4,8,16,32,64,128,256],[res["val_loss"].iloc[-1] for res in resultsArray],marker="o")
    #plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.xlabel("Batch size",fontsize=14)
    plt.ylabel("val error",fontsize=14)
    plt.savefig(os.path.join("batchSizeAnalError.pdf"))
    plt.show()
    plt.figure(figsize=(4.5,3))
    #plt.plot([1,2,4,8,16,32,64,128,256],[1-res["val_accuracy"].iloc[-1] for res in resultsArray],marker="o",color="C2")
    plt.plot([1,2,4,8,16,32,64,128,256],[res["val_loss"].iloc[-1] for res in resultsArray],marker="o")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.xlabel("Batch size",fontsize=14)
    plt.ylabel("val loss",fontsize=14)
    plt.savefig("batchSizeAnalLoss.pdf")
    plt.tight_layout()
    plt.show()


# %%
