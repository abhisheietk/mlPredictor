import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from tensorflow import estimator
import pandas as pd
tf.reset_default_graph()

def normalize(df):    
    return (df - df.min()) / (df.max() - df.min())


def denormalize(df,norm_data):    
    return (norm_data * (df.max() - df.min())) + df.min()


def preProcessForTrain(sensor_no):
    sensor=str(sensor_no)
    df = pd.read_csv('./data/finalData/'+sensor+'.csv')
    df['out']=df['out']*1
    df = df.dropna(inplace=False)
    msk = np.random.rand(len(df)) < 0.6
    df_train = df[msk]    
    df_test = df[~msk]
    X_train = normalize(df_train.drop(['out'],axis=1)).values
    y_train = df_train['out'].values
    X_test = normalize(df_test.drop(['out'],axis=1)).values
    y_test = df_test['out'].values
    return X_train, y_train, X_test, y_test


def preProcessForTest(sensor_no):
    sensor=str(sensor_no)
    df = pd.read_csv('./data/finalData/'+sensor+'.csv')
    df['out']=df['out']*1
    df = df.dropna(inplace=False)
    msk = np.random.rand(len(df)) < 1
    df_test = df[msk]
    X_test = normalize(df_test.drop(['out'],axis=1)).values
    y_test = df_test['out'].values
    return X_test, y_test


def DNNmodel():
    feat_cols=[tf.feature_column.numeric_column('x', shape=[960])]
    return tf.estimator.DNNClassifier(feature_columns=feat_cols, 
                                      n_classes=2,
                                      hidden_units=[2000,2000,2000,2000,2000,2000], 
                                      activation_fn=tf.nn.relu,
                                      optimizer=tf.train.GradientDescentOptimizer(0.0001),
                                      model_dir='./model')


def training(X_train, y_train):
    deep_model=DNNmodel()
    input_fn = estimator.inputs.numpy_input_fn(x={'x':X_train}, 
                                           y=y_train,
                                           shuffle= True,
                                           num_epochs=5000,
                                           batch_size=100)
    return deep_model.train(input_fn=input_fn, steps=50000)


def evaluate(X_test, y_test):   
    input_fn_eval = estimator.inputs.numpy_input_fn( x = {'x':X_test},
                                                y =  y_test,
                                                shuffle = False)
    deep_model=DNNmodel()
    
    return dict(deep_model.evaluate(input_fn=input_fn_eval))



def predict(X_test):
    
    input_fn_eval = estimator.inputs.numpy_input_fn( x = {'x':X_test},
                                                shuffle = False)
    deep_model = DNNmodel()
    preds=list(deep_model.predict(input_fn=input_fn_eval))
    predictions = [p['class_ids'][0] for p in preds]
    pred = np.asarray(predictions)
    return pred    


def checkAccuracy(pred, y_test):
    f = pred == y_test
    T = len(f[f == True])
    F = len(f[f == False])
    error = F/(T+F) * 100
    return T, F, error 


def preProcessForMergeTrain(filepath):
    df = pd.read_csv(filepath)
    df['out']=df['out']*1
    df = df.dropna(inplace=False)
    msk = np.random.rand(len(df)) < 0.6
    df_train = df[msk]    
    df_test = df[~msk]
    X_train = df_train.drop(['out'],axis=1).values
    y_train = df_train['out'].values
    X_test = df_test.drop(['out'],axis=1).values
    y_test = df_test['out'].values
    return X_train, y_train, X_test, y_test


def preProcessForMergeTest(filepath):
    df = pd.read_csv(filepath)
    df['out']=df['out']*1
    df = df.dropna(inplace=False)
    df_test = df
    X_test = df_test.drop(['out'],axis=1).values
    y_test = df_test['out'].values
    return X_test, y_test

