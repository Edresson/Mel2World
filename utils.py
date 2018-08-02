import numpy as np
import pyworld as pw
import librosa
from hyperparams import Hyperparams as hp

import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import tensorflow as tf

def world_features_to_one_tensor(f0,sp,ap):
    return np.column_stack((np.column_stack((np.array(f0),np.array(sp))),np.array(ap)))

def tensor_to_world_features(tensor):
    f0=[]
    sp=[]
    ap = []
    for i in range(len(tensor)):
        
        sp.append(np.array(tensor[i][1:514]))
        ap.append(np.array(tensor[i][514:]))
        f0.append(np.array(tensor[i][0]))
    return np.array(f0),np.array(sp),np.array(ap) 

def wav2world(wavfile):
    x, fs = librosa.load(wavfile, dtype=np.float64)
    return pw.wav2world(x, fs)

def world2wav(f0,sp,ap,fs):
    return pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)

def unpad_world(mel,world):
    '''
    mel: input Mel spectrogram not pad.
    world: world features predict tensor.
    return unpad tensor world features.
    '''     

    t = mel.shape[0]
    t2=world.shape[0]
    
    num_paddings = hp.padf - (t % hp.padf) if t % hp.padf != 0 else 0
    unpad = world[:t2-num_paddings]
        
    return unpad
    
def pad_mel_world(mel,world=None):
    '''
    mel: Mel spectrogram.
    world: world features for train. if None ignore pad for world features
    return Mel spectrogram e world features. if world==None return same Mel Spectrogram.
    '''    
    t = mel.shape[0]
    
    num_paddings = hp.padf - (t % hp.padf) if t % hp.padf != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    if(world is not None):
        world = np.pad(world, [[0, num_paddings], [0, 0]], mode="constant")
        return mel,world
    return mel



def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
 
    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)

    return mel


def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    mel = get_spectrograms(fpath)
    
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    # Reduction
    mel = mel[::hp.r, :]
    return mel

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')


def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
