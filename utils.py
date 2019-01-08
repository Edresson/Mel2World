import numpy as np
import pyworld as vocoder
import librosa
from hyperparams import Hyperparams as hp
import soundfile as sf
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import tensorflow as tf

int16_max = 32768.0

def f0_normalize(x):
	return np.log(np.where(x == 0.0, 1.0, x)).astype(np.float32)

def f0_denormalize(x):
	return np.where(x == 0.0, 0.0, np.exp(x.astype(np.float64)))

def sp_normalize(x):
	sp = int16_max * np.sqrt(x)
	return pysptk.sptk.mcep(sp.astype(np.float32), order=hp.num_mgc - 1, alpha=hp.mcep_alpha,
				maxiter=0, threshold=0.001, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

def sp_denormalize(x):
	sp = pysptk.sptk.mgc2sp(x.astype(np.float64), order=hp.num_mgc - 1,
				alpha=hp.mcep_alpha, gamma=0.0, fftlen=hp.n_fft)
	return np.square(sp / int16_max)

def ap_normalize(x):
	return x.astype(np.float32)

def ap_denormalize(x, lf0):
	for i in range(len(lf0)):
		x[i] = np.where(lf0[i] == 0, np.zeros(x.shape[1]), x[i])
	return x.astype(np.float64)


def world_features_to_one_tensor(f0,sp,ap):
    return np.column_stack((np.column_stack((np.array(f0),np.array(sp))),np.array(ap)))

def tensor_to_world_features(tensor):
    f0=[]
    sp=[]
    ap = []
    sp_factor = hp.num_mgc+1
    for i in range(len(tensor)):
        f0.append(np.array(tensor[i][0]))
        sp.append(np.array(tensor[i][1:sp_factor]))
        ap.append(np.array(tensor[i][sp_factor:]))
        
    return np.array(f0),np.array(sp),np.array(ap)

def wav2world(wavfile):
    wav, fs = sf.read(wavfile)
    f0,sp,ap=vocoder.wav2world(wav,fs , hp.n_fft, ap_depth=hp.num_bap)
    # feature normalization
    lf0 = f0_normalize(f0)
    mgc = sp_normalize(sp)
    bap = ap_normalize(ap)
    return np.array(world_features_to_one_tensor(lf0,mgc,bap))


def world2wav(lf0, mgc, bap):
	lf0 = np.where(lf0 < 1, 0.0, lf0)
	f0 = f0_denormalize(lf0)
	sp = sp_denormalize(mgc)
	ap = ap_denormalize(bap, lf0)
	print('features denomalize',lf0.shape,sp.shape,ap.shape)
	wav = vocoder.synthesize(f0, sp, ap,hp.sr_dataset)
	return wav

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
