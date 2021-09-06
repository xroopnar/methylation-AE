import pandas as pd 
import sys
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from hyperopt import Trials,STATUS_OK,tpe

import hyperas
from hyperas import optim
from hyperas.distributions import choice,uniform
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
import utils

sys.path.insert(0,"/home/xiavan/gitlab/mana/")
import mana
import utils as mutils

def make_model():
    encoder_input = layers.Input(48242)
    encoded = layers.Dense(1024,activation="sigmoid")(encoder_input)
    decoded = layers.Dense(48242,activation="sigmoid")(encoded)
    autoencoder = keras.Model(inputs=encoder_input,outputs=decoded)
    autoencoder.compile(optimizer="adam",loss="binary_crossentropy")
    return(autoencoder)

def tissue_model():
    metrics = ["accuracy","AUC"]
    encoder_input = layers.Input(48242)
    encoded = layers.Dense(2048,activation="sigmoid")(encoder_input)
    #encoded = layers.Dense(1024,activation="sigmoid")(encoded)
    #decoded = layers.Dense(48242,activation="sigmoid")(encoded)
    predictor = layers.Dense(10,activation="softmax")(encoded)
    autoencoder = keras.Model(inputs=encoder_input,outputs=predictor)
    autoencoder.compile(optimizer="adam",loss="categorical_crossentropy",metrics=metrics)
    return(autoencoder)

#age model fails 
def age_model():

    opt = keras.optimizers.Adam(learning_rate=0.001)
    encoder_input = layers.Input(48242)
    #encoded = layers.Dense(1024,activation="leakyrelu")(encoder_input)
    encoded = layers.Dense(1024)(encoder_input)
    encoded = layers.LeakyReLU(alpha=0.05)(encoded)
    #decoded = layers.Dense(48242,activation="sigmoid")(encoded)
    predictor = layers.Dense(1,activation="linear")(encoded)
    autoencoder = keras.Model(inputs=encoder_input,outputs=predictor)
    #autoencoder.compile(optimizer="sgd",loss="mean_absolute_error")
    autoencoder.compile(optimizer=opt,loss="mean_absolute_error")
    return(autoencoder)

def go_model(n=16541):
    metrics = ["accuracy","AUC"]
    encoder_input = layers.Input(n)
    #encoded = layers.Dense(1024,activation="sigmoid")(encoder_input)
    encoded = layers.Dense(8192,activation="sigmoid")(encoder_input)
    encoded = layers.Dense(4096,activation="sigmoid")(encoded)
    #encoded = layers.Dense(2048,activation="sigmoid")(encoded)
    #    #encoded = layers.Dense(16,activation="sigmoid")(encoder_input)
    
    #decoded = layers.Dense(48242,activation="sigmoid")(encoded)
    predictor = layers.Dense(12323,activation="softmax")(encoded)
    #predictor = layers.Dense(10,activation="softmax")(decoded)
    autoencoder = keras.Model(inputs=encoder_input,outputs=predictor)
    #autoencoder.compile(optimizer="adam",loss="binary_crossentropy",metrics=metrics)
    autoencoder.compile(optimizer="adam",loss="categorical_crossentropy",metrics=metrics)
    return(autoencoder)

def run_go(X=utils.load_gb(),epochs=20):
    #go matrix, term vs gene
    y = utils.load_go()
    print(y.shape)
    #gb means vs samples
    #X = utils.load_gb()
    print(X.shape)
    X,y = X.align(y,axis=1,join='inner')
    X,y =X.T,y.T
    train_X,val_X,train_y,val_y = train_test_split(X,y,shuffle=True,test_size=0.2)
    #train_X,val_X,train_y,val_y = [x.T for x in [train_X,val_X,train_y,val_y]]
    print("final dims:")
    [print(x.shape) for x in [train_X,val_X,train_y,val_y]]
    model = go_model(n=train_X.shape[1])
    model.fit(x=train_X,y=train_y,validation_data=(val_X,val_y),shuffle=True,epochs=epochs,verbose=2)
    return(model,val_X,val_y)

def run_tissue():
    X = utils.fetch_data()
    meta = utils.fetch_meta()
    X,meta = X.align(meta,axis=0,join="left")
    y = pd.get_dummies(meta.TissueName)
    val_x = X.sample(frac=0.2,random_state=4433)
    val_y = y.loc[val_x.index]
    train_x = X.drop(val_x.index)
    train_y = y.loc[train_x.index]
    print(train_x.shape,train_y.shape,val_x.shape,val_y.shape)
    train_x,val_x = train_x.values,val_x.values
    print("sample dims:",X.shape)
    print("label dims:",y.shape)

    #model = make_model()
    #model.fit(x=train_x,y=train_x,validation_data=(val_x,val_x),epochs=15,verbose=2,shuffle=True)
    model = tissue_model()
    model.fit(x=train_x,y=train_y,validation_data=(val_x,val_y),epochs=50,verbose=2,shuffle=True)
    return(model,val_x,val_y)

def run_age():
    X = utils.fetch_data()
    meta = utils.fetch_meta()
    meta = meta.dropna(subset=["Age"])
    X,meta = X.align(meta,axis=0,join="left")
    y = meta.Age
    val_x = X.sample(frac=0.2,random_state=4433)
    val_y = y.loc[val_x.index]
    train_x = X.drop(val_x.index)
    train_y = y.loc[train_x.index]
    print(train_x.shape,train_y.shape,val_x.shape,val_y.shape)
    train_x,val_x = train_x.values,val_x.values
    print("sample dims:",X.shape)
    print("label dims:",y.shape)

    #model = make_model()
    #model.fit(x=train_x,y=train_x,validation_data=(val_x,val_x),epochs=15,verbose=2,shuffle=True)
    model = age_model()
    model.fit(x=train_x,y=train_y,validation_data=(val_x,val_y),epochs=50,verbose=2,shuffle=True)
    return(model)

def hypermodel_data():
    #go matrix, term vs gene
    y = utils.load_go()
    print(y.shape)
    #gb means vs samples
    X = utils.load_gb()
    print(X.shape)
    X,y = X.align(y,axis=1,join='inner')
    X,y =X.T,y.T
    train_X,val_X,train_y,val_y = train_test_split(X,y,shuffle=True,test_size=0.2)
    return(train_X,train_y,val_X,val_y)

def main():
    #run_age()
    #run_tissue()
    #print("genebody mean GO prediction:")
    #run_hypermodel()
    run_go()
    #print("TSS1500 mean GO prediction:")
    #run_go(X=utils.load_tss())

if __name__ == "__main__":
    main()
