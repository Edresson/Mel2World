# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

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
def load_data():
    '''Loads data
    '''
    fpaths=os.listdir('mels\\')
    audio_lenghts = []
    for path in fpaths:
        audio_lenghts.append(librosa.get_duration(filename=os.path.join(hp.data_dir,path.replace('.npy','.wav'))))    
    return fpaths, audio_lenghts



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

