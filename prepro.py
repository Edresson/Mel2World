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
    try:
        #print(i,'de: ',data_lenght)
        mel= np.array(load_spectrograms(hp.data_dir+'/'+data_list[i])) # mel spectrogram
        world=wav2world(hp.data_dir+'/'+data_list[i])
        num_padding = mel.shape[0]*8 - world.shape[0] 
        world = np.pad(world, [[0, num_padding], [0, 0]], mode="constant")
        np.save("mels/{}".format(data_list[i].replace("wav", "npy")), mel)
        np.save("worlds/{}".format(data_list[i].replace("wav", "npy")), world)
    except:
        continue

print('preprocessing ok !!!')

