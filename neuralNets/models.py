import numpy as np
import tensorflow as tf

# regular models 

def oneLayer1():
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1")(inputs)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def oneLayer2(activation="relu",penalization=0.0, size=100):
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
    x = tf.keras.layers.Dense(size,activation=activation,name="Dense_Layer_1",kernel_regularizer=tf.keras.regularizers.l1(l=penalization))(inputs)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def twoLayers(activation=["relu","relu"],penalization=[0.0,0.0],size=[100,100]):
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
    x = tf.keras.layers.Dense(size[0],activation="relu",name="Dense_Layer_1")(inputs)
    x = tf.keras.layers.Dense(size[1],activation="relu",name="Dense_Layer_2")(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def twoLayersDropout(activation=["relu","relu"],penalization=[0.0,0.0],size=[100,100],dropout=0.5):
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
    x = tf.keras.layers.Dense(size[0],activation="relu",name="Dense_Layer_1")(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(size[1],activation="relu",name="Dense_Layer_2")(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

def fourLayers(activation = ["relu"]):
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
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
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
    x = tf.keras.layers.Dropout(input_dropout)(inputs)
    x = tf.keras.layers.Dense(size[0],activation=activation[0],name=layerNames[0],kernel_regularizer=tf.keras.regularizers.l1(l=penalization[0]))(x)
    x = tf.keras.layers.Dropout(dropout[0])(x)
    for i in range(len(activation)-1):
        x = tf.keras.layers.Dense(size[i+1],activation=activation[i+1],name=layerNames[i+1],kernel_regularizer=tf.keras.regularizers.l1(l=penalization[i+1]))(x)
        x = tf.keras.layers.Dropout(dropout[i+1])(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")


def unevenLayeredModel(layers  = 3 , l = 0.0, mode = "lin"):
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
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
    inputs = tf.keras.Input(shape=(28*28),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv1")

def convMod2():
    inputs = tf.keras.Input(shape=(28*28),name="Input")
    x = tf.keras.layers.Reshape((28,28,1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=(28,28,1))(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100,activation="relu",name="Dense_Layer_1",kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(x)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="Conv2")

def convMod3():
    inputs = tf.keras.Input(shape=(28*28),name="Input")
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
    inputs = tf.keras.Input(shape=(28*28),name="Input")
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
    inputs = tf.keras.Input(shape=(28*28),name="Input")
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

#logistic regression model

def logReg():
    inputs = tf.keras.Input(shape=(28*28,),name="Input")
    outputs = tf.keras.layers.Dense(10,activation="linear",name="Logit_Probs")(inputs)
    return tf.keras.Model(inputs=inputs,outputs=outputs,name="initModel")

# compilation functions

def basicCompile(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

def interestingCompile(model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy',
              tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
