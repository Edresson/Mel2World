# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import sys
import librosa
def load_data(mode = 'train'):
    '''Loads data
    '''
    fpaths=os.listdir('mels\\')
    return_fpaths = []
    if mode == 'train':
        audio_lenghts = []
        for path in fpaths:
            file_name,_ = path.split('.')
            #print(file_name)
            file_id = int(file_name.split('-')[1])
            #ignore test
            if file_id >=5654 and file_id <=5674:
                print('file:',path,' ignored ')
                continue
            return_fpaths.append(path)
            audio_lenghts.append(librosa.get_duration(filename=os.path.join(hp.data_dir,path.replace('.npy','.wav'))))    
        return return_fpaths, audio_lenghts
    else:
        # test data
        for path in fpaths:
            file_name,_ = path.split('.')
            file_id = int(file_name.split('-')[1])
            if file_id >=5654 and file_id <=5674:
                return_fpaths.append(path)

        return return_fpaths

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, audio_lengths = load_data() # list
        maxlen, minlen = int(max(audio_lengths)), int(min(audio_lengths))

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, audio_length = tf.train.slice_input_producer([fpaths,audio_lengths], shuffle=True)


        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                fname = fname.decode("utf8")
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                world = "worlds/{}".format(fname.replace("wav", "npy"))
                mels = np.load(mel)
                worlds = np.load(world)
                return fname, mels, worlds

            fname, mel, world = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            print(' Please Run Prepo.py !!')

        # Add shape information
        fname.set_shape(())
        mel.set_shape((None, hp.n_mels))
        world.set_shape((None, hp.num_lf0+hp.num_mgc+hp.num_bap))

        # Batching
        mels, worlds = tf.train.batch([mel, world],batch_size=hp.B,num_threads=4,capacity=hp.B*4, dynamic_pad=True)

    return mels, worlds,  num_batch

