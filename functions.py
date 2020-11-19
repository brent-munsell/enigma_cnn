import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import tensorflow.keras.backend as K
import os
import pickle
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve 
import statistics as stat

def make_opts():
    opts = {'lr' : random.choice(np.arange(0.0001, 0.00101, 0.00005)), 'mx_epochs' : random.choice(np.arange(40, 201, 20)), 'val_freq' : random.choice(np.arange(10, 81, 10)).item(), 'F1S' : random.choice(np.asarray([3, 5, 10, 15, 20, 25, 30, 32, 35, 40])), 
            'F1N' : random.choice(np.asarray([10, 20, 30, 40])), 'F2S' : random.choice(np.asarray([3, 5, 10, 15, 20, 25])), 'F2N' : random.choice(np.asarray([10, 20, 30, 40])), 'F3S' : random.choice(np.asarray([3, 5, 10, 15])), 'F3N' : random.choice(np.asarray([10, 20, 30, 40])), 'batch_size' : random.choice(np.asarray([8, 16, 32, 64]))}
    return opts

def show_slice(image, subject_type, jj):
    plt.title('{0:s} slice {1:d}'.format(subject_type, jj))
    plt.imshow(image, cmap="gray")
    return plt

def cnn_network(opts):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(opts['siz'], opts['siz'], 1), name = 'input'))
    model.add(layers.Conv2D(opts['F1N'], (opts['F1S'], opts['F1S']), padding = 'same', name = 'conv_1', activation = 'relu'))
    model.add(layers.BatchNormalization(name = 'BN_1'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(opts['F2N'], (opts['F2S'], opts['F2S']), strides = 2, padding = 'same', name = 'conv_2', activation = 'relu'))
    model.add(layers.BatchNormalization(name = 'BN_2'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(opts['F3N'], (opts['F3S'], opts['F3S']), padding = 'same', name = 'conv_3', activation = 'relu'))
    model.add(layers.BatchNormalization(name = 'BN_3'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', name = 'dense_1'))
    model.add(layers.Dense(2, name = 'dense_2', activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=opts['lr'], momentum = 0.9), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])  
    return model

def split(X, y, num_fold):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, stratify = y)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    T = {'xtest': X_test, 'ytest': y_test}
    S = {'xtrain' : [], 'ytrain' : [], 'xval' : [], 'yval' : []}
    kf = StratifiedKFold(n_splits=num_fold, shuffle = True)
    for train_index, test_index in kf.split(X_train, y_train):
        S['xtrain'].append(X_train[train_index]) 
        S['xval'].append(X_train[test_index])
        S['ytrain'].append(y_train[train_index])
        S['yval'].append(y_train[test_index])
    return T, S

def activation_weights(net, im, list_layers):
    H = [None] * len(list_layers)
    for i in range(0, len(list_layers)):
        q = im
        q = np.expand_dims(q, axis=0)
        activations = K.function([net.layers[0].input], net.get_layer(name = list_layers[i]).output)
        act1 = activations([q, 0])
        imgSize = im.shape[0:2]
        _, _, _, maxValueIndex = np.where(act1 >= act1.max())
        act1chMax = act1[:, :, :, maxValueIndex]
        act1chMax = (act1chMax - act1chMax.min()) / act1chMax.max()
        H[i] = act1chMax
    return H
 
def process_results(opts_dir, net_dir): 
    files = list(os.scandir(opts_dir))
    files2 = list(os.scandir(net_dir))
    C = {'ac' : np.zeros((1000), dtype = int), 'ppv' : np.zeros((1000), dtype = int), 'npv' : np.zeros((1000), dtype = int), 'cm' : np.zeros((1000, 2, 2), dtype = int),
        'spc' : np.zeros((1000), dtype = int), 'sen' : np.zeros((1000), dtype = int), 'auc' : np.zeros((1000), dtype = int),
        'ax' : np.zeros((1000, 3), dtype = int), 'ay' : np.zeros((1000, 3), dtype = int), 'cnt' : 1}
    A = [[]] * 3
    list_layers = ['conv_1', 'conv_2', 'conv_3']
    
    for i in range(0, len(files)):
        if files[i].is_file():
            f = open(files[i], "rb")
            d = pickle,load(f) 
            f.close()
            
            tt = d['T']['ytest'].flatten() 
            
            for j in d['ypred']:
                pp = j.flatten() 
                cm = tf.math.confusion_matrix(tt, pp).numpy()
                ax, ay, T = roc_curve(tt, pp)
                auc = roc_auc_score(tt, pp)
                
                C['cm'][C['cnt']] = cm
                C['ac'][C['cnt']] = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
                C['ppv'][C['cnt']] = cm[0, 0] / np.sum(cm[0, :])
                C['npv'][C['cnt']] = cm[1, 1] / np.sum(cm[1, :])
                C['sen'][C['cnt']] = cm[0, 0] / np.sum(cm[:, 0])
                C['spc'][C['cnt']] = cm[1, 1] / np.sum(cm[:, 1])
                C['auc'][C['cnt']] = auc
                C['ax'][C['cnt']] = ax
                C['ay'][C['cnt']] = ay
                C['cnt'] = C['cnt'] + 1
                
            for j in os.scandir(files2[i]):
                net = tf.keras.models.load_model(j)
                for tt in d['T']['xtest']:
                    H = activation_weights(j, tt, list_layers)
                    for k in range(0, len(A)):
                        if (len(A[k]) == 0):
                            A[k] = H[k]
                        else: 
                            A[k] = A[k] + H[k]
                            
    for i in range(0, len(A)):
        A[i] = (A[i] - A[i].min()) / A[i].max()  
    C['ac_avg'] = C['ac'].mean(axis = 0)
    return C, A

def show_activations(A):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fix2, ax3 = plt.subplots()
    ax1.set_title('conv_1 activations')
    ax1.imshow(A[0][0, :, :, 0], cmap="hot")
    ax2.set_title('conv_2 activations')
    ax2.imshow(A[1][0, :, :, 0], cmap="hot")
    ax3.set_title('conv_3 activations')
    ax3.imshow(A[2][0, :, :, 0], cmap="hot")
    return fig1, fig2, fig3

def plot_roc(C):
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(C['ax'], C['ay'], label='area = {:.3f}'.format(C['auc']))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    return plt
    
def grid_results(folder):
    P = []
    A = []
    F = []
    
    files = os.scandir(folder)            
    for path in files:
        if path.is_file():
            f = open(path, "rb")
            opts = pickle.load(f) 
            f.close()
            p = [None] * 10
            fields = list(opts.keys())
            for j in range(0, 10):
                p[j] = opts[fields[j]]
            P.append(p)
            A.append(opts['acc'])
            F.append(f.name)
            f.close()
    D = {'P' : np.asarray(P), 'A' : np.asarray(A), 'F' : np.asarray(F)}
    opts = optimal_grid_parameters(D)
    return D, opts

def optimal_grid_parameters(D):
    idx = np.argmax(D['A'].mean(axis= 1) - D['A'].std(axis=1))
    P = D['P'][idx, :]
    opts = {'lr' : P[0], 'mx_epochs' : P[1].astype(int), 'val_freq' : P[2].astype(int).item(), 'F1S' : P[3].astype(int), 
            'F1N' : P[4].astype(int), 'F2S' : P[5].astype(int), 'F2N' : P[6].astype(int), 'F3S' : P[7].astype(int), 'F3N' : P[8].astype(int), 'batch_size' : P[9].astype(int)}
    return opts

def process_slice(opts_dir, net_dir):
    files = list(os.scandir(opts_dir))
    files2 = list(os.scandir(net_dir))
    C = []
    cnt = 0
    pslice = [0] * 156        
    list_layers = ['conv_1', 'conv_2', 'conv_3']
    for i in range(0, len(files)):
        if files[i].is_file():
            f = open(files[i], "rb")
            d = pickle,load(f) 
            f.close()
            A = [[]] * 3
            tt = d['T']['ytest'].flatten() 
            C.append({'slice' : d['jj'], 'cm' : np.zeros((2, 2), dtype = int),
                      'ac' : np.zeros(len(d['ypred']), dtype = int), 'ppv' : np.zeros(len(d['ypred']), dtype = int),
                      'npv' : np.zeros(len(d['ypred']), dtype = int), 'A' : []})
            for j in range(0, len(d['ypred'])):
                pp = d['ypred'][j].flatten() 
                cm = tf.math.confusion_matrix(tt, pp).numpy()
                
                C[cnt]['cm'] = C[cnt]['cm'] + cm
                C[cnt]['ac'][j] = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
                C[cnt]['ppv'][j] = cm[0, 0] / np.sum(cm[0, :])
                C[cnt]['npv'][j] = cm[1, 1] / np.sum(cm[1, :])
            
            for j in os.scandir(files2[i]):
                net = tf.keras.models.load_model(j)
                H = activation_weights(j, d['T']['xtest'][0], list_layers)
                for k in range(0, len(A)):
                    if (len(A[k]) == 0):
                        A[k] = H[k]
                    else: 
                        A[k] = A[k] + H[k]
            for i in range(0, len(A)):
                A[i] = (A[i] - A[i].min()) / A[i].max()  
            
            pslice[C[cnt]['slice']] = 1
            C[cnt]['A'] = A
            cnt = cnt + 1
            d = []
    return C, pslice

def slice_results(folder):
    P = []
    A = []
    F = []
    S = []
    
    files = os.scandir(folder)            
    for path in files:
        if path.is_file():
            f = open(path, "rb")
            opts = pickle.load(f) 
            f.close()
            p = [None] * 10
            fields = list(opts.keys())
            for j in range(0, 10):
                p[j] = opts[fields[j]]
            P.append(p)
            A.append(opts['acc'])
            F.append(f.name)
            S.append(opts['jj'])
            f.close()
    D = {'P' : np.asarray(P), 'A' : np.asarray(A), 'F' : np.asarray(F), 'S' : np.asarray(S)}
    opts = optimal_slice(D)
    return D, opts

def optimal_slice(D):
    idx = np.argmax(D['A'].mean(axis= 1) - D['A'].std(axis=1))
    P = D['P'][idx, :]
    S = D['S'][idx]
    opts = {'lr' : P[0], 'mx_epochs' : P[1].astype(int), 'val_freq' : P[2].astype(int).item(), 'F1S' : P[3].astype(int), 
            'F1N' : P[4].astype(int), 'F2S' : P[5].astype(int), 'F2N' : P[6].astype(int), 'F3S' : P[7].astype(int), 'F3N' : P[8].astype(int), 'batch_size' : P[9].astype(int), 'jj' : S.astype(int), 'P' : './Data/P_left_sm.mat', 'C' : './Data/C_sm.mat', 'res_dir' : './optimal_slice'}
    return opts