from __future__ import print_function

from utils import load_spectrograms,wav2world,world_features_to_one_tensor
import os
import numpy as np
from hyperparams import Hyperparams as hp

if not os.path.exists("data"): os.mkdir("data")

X =[]
Y=[]
XTest=[]
YTest=[]

# data directory
data_list=os.listdir(hp.data_dir)
data_lenght=len(data_list)

for i in range(0,int(data_lenght/2)):
    X.append(load_spectrograms(hp.data_dir+'/'+data_list[i])) # mel spectrogram
    f0,sp,ap=wav2world(hp.data_dir+'/'+data_list[i])
    Y.append(world_features_to_one_tensor(f0,sp,ap)) # world features
    
for i in range(int(data_lenght/2) +1,data_lenght):
    XTest.append(load_spectrograms(hp.data_dir+'/'+data_list[i])) # mel spectrogram
    f0,sp,ap=wav2world(hp.data_dir+'/'+data_list[i])
    YTest.append(world_features_to_one_tensor(f0,sp,ap)) # world features
    
np.save("data/X_train.npy", X)
np.save("data/Y_train.npy", Y)

np.save("data/X_test.npy", XTest)
np.save("data/Y_test.npy", YTest)

