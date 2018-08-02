import numpy as np
from hyperparams import Hyperparams as hp

from utils import tensor_to_world_features,world2wav,unpad_world,pad_mel_world
import librosa

X = np.load('data/X_train.npy')
Y = np.load('data/Y_train.npy')


origMel=X[3]
origWorld=Y[3]
mel,world=pad_mel_world(origMel,origWorld)
print(world.shape)
world=unpad_world(origMel,world)
print(world.shape)
f0,sp,ap= tensor_to_world_features(world)
f01,sp1,ap1= tensor_to_world_features(origWorld)
y1=world2wav(f01,sp1,ap1,hp.sr)
y = world2wav(f0,sp,ap,hp.sr)
print(y.shape,y1.shape)

librosa.output.write_wav('2.wav', y1, hp.sr)
librosa.output.write_wav('1.wav', y, hp.sr)
