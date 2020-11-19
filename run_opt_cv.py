from functions import make_opts, cnn_network, split
import pickle
import numpy as np
import h5py
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("slice", type = int)
parser.add_argument("dt", type = str)
args = parser.parse_args()
jj = args.slice
dt = args.dt

os.mkdir('./run_opt_cv_nets/nets_{0:d}'.format(dt))

f = open('./optimal_slice/opt_slice_opts.pkl', "rb")
opts = pickle.load(f) 
f.close()

path_P = opts['P']
feature_P =h5py.File(path_P, 'r')
p_t1 = np.array(feature_P['P']['T1']) 
print('loaded patient image data set ...') 

path_P = opts['C']
feature_P=h5py.File(path_P, 'r') 
c_t1 = np.array(feature_P['C']['T1']) 
print('loaded control image data set ...') 

opts.update({'siz': 142, 'num_fold': 10, 'tt': 11, 'zz': 11})

c_t2 = np.moveaxis(c_t1, 0, -1)
p_t2 = np.moveaxis(p_t1, 0, -1)

Coronal_controls = np.fliplr(np.flipud(c_t2[opts['zz']:opts['zz']+opts['siz'], opts['jj'], opts['tt']:opts['tt']+opts['siz'], :]))
Coronal_patients = np.fliplr(np.flipud(p_t2[opts['zz']:opts['zz']+opts['siz'], opts['jj'], opts['tt']:opts['tt']+opts['siz'], :]))

Coronal_controls = np.moveaxis(Coronal_controls, -1, 0)
Coronal_patients = np.moveaxis(Coronal_patients, -1, 0)

X = np.concatenate((Coronal_controls, Coronal_patients), axis=0)
y = np.concatenate((np.ones(Coronal_controls.shape[0], dtype = int), np.zeros(Coronal_patients.shape[0], dtype = int)), axis=0)

T, S = split(X, y, opts['num_fold'])
opts.update({'T': T, 'S': S, 'ypred': [], 'ypredr':[], 'acc': []})

for j in range(0, len(S['ytrain'])):
    net = cnn_network(opts)
    net.fit(S['xtrain'][j], S['ytrain'][j], epochs=opts['mx_epochs'], shuffle=True, validation_freq=opts['val_freq'], validation_data=(S['xval'][j], S['yval'][j]), batch_size=opts['batch_size']) 
    ypred = np.argmax(net.predict(T['xtest']), axis=-1)
    acc = 0
    for i in range(0, len(ypred)):
        if (ypred[i] == T['ytest'][i][0]): 
            acc = acc + 1
    opts['acc'].append(acc/len(ypred))
    opts['ypred'].append(ypred)
    
    now = datetime.now() # current date and time
    dt2 = now.strftime("%m-%d-%y-%H-%M-%S-%f")
    net.save('./run_opt_cv_nets/nets_{0:d}/net_{1:s}'.format(dt, dt2))

f = open('./run_opt_cv_opts/cv_opts_{0:d}.pkl'.format(dt),"wb")
pickle.dump(opts, f)
f.close()