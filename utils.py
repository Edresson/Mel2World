import numpy as np
import pyworld as pw
import librosa
from hyperparams import Hyperparams as hp

def world_features_to_one_tensor(f0,sp,ap):
    return np.column_stack((np.column_stack((np.array(f0),np.array(sp))),np.array(ap)))

def tensor_to_worl_features(tensor):
    f0=[]
    sp=[]
    ap = []
    for i in range(len(tensor)):
        
        sp.append(tensor[i][1:514])
        ap.append(tensor[i][514:])
        f0.append(tensor[i][0])
    return np.array(f0),np.array(sp),np.array(ap) 

def wav2world(wavfile):
    x, fs = librosa.load(wavfile, dtype=np.float64)
    return pw.wav2world(x, fs)

def world2wav(f0,sp,ap):
    pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
    
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


