# -*- coding: utf-8 -*-
#/usr/bin/python3
from __future__ import print_function

from utils import load_spectrograms,wav2world,world_features_to_one_tensor
import os
import numpy as np
from hyperparams import Hyperparams as hp
import tqdm

if not os.path.exists("mels"): os.mkdir("mels")
if not os.path.exists("worlds"): os.mkdir("worlds")

X =[]
Y=[]
XTest=[]
YTest=[]

# data directory
data_list=os.listdir(hp.data_dir)
data_lenght=len(data_list)

for i in tqdm.tqdm(range(0,data_lenght)):
    mel= np.array(load_spectrograms(hp.data_dir+'/'+data_list[i])) # mel spectrogram
    f0,sp,ap=wav2world(hp.data_dir+'/'+data_list[i])
    world=np.array(world_features_to_one_tensor(f0,sp,ap)) # world features
    np.save("mels/{}".format(data_list[i].replace("wav", "npy")), mel)
    np.save("worlds/{}".format(data_list[i].replace("wav", "npy")), world)

'''
for i in tqdm.tqdm(range(int(data_lenght/2) +1,data_lenght)):
    XTest.append(np.array(load_spectrograms(hp.data_dir+'/'+data_list[i]))) # mel spectrogram
    f0,sp,ap=wav2world(hp.data_dir+'/'+data_list[i])
    YTest.append(np.array(world_features_to_one_tensor(f0,sp,ap))) # world features

np.save("data/X_test.npy", np.array(XTest))
np.save("data/Y_test.npy", np.array(YTest))'''

