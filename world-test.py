import soundfile as sf
import pysptk
import numpy as np
import pyworld as vocoder
import os
import librosa

from hyperparams import Hyperparams as hp

int16_max = 32768.0

def get_spectrograms(path):
    '''
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
    '''
    y, _ =librosa.load(path,sr=hp.sr)
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
    
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    # Reduction
    mel = mel[::hp.r, :]
    return mel

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

def synthesize(lf0, mgc, bap,sample_rate):
	lf0 = np.where(lf0 < 1, 0.0, lf0)
	f0 = f0_denormalize(lf0)
	sp = sp_denormalize(mgc)
	ap = ap_denormalize(bap, lf0)
	print('features denomalize',lf0.shape,sp.shape,ap.shape)
	wav = vocoder.synthesize(f0, sp, ap,sample_rate)
	return wav

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

audiolist= ['sample-208.wav','chunk4.wav']
for file in  audiolist:
    #wav, fs =librosa.load(file,sr=hp.sr, dtype=np.float64)
    wav, fs = sf.read(file)
    sample_rate = fs
    f0, sp, ap = vocoder.wav2world(wav,sample_rate , hp.n_fft, ap_depth=hp.num_bap)
    mel = get_spectrograms(file)
    #print('features ',file,':',f0.shape,sp.shape,ap.shape)
    print('mel features ',file,':', mel.shape)

    # feature normalization
    lf0 = f0_normalize(f0)
    mgc = sp_normalize(sp)
    bap = ap_normalize(ap)
    #print('features normalized ',file,':',lf0.shape,mgc.shape,bap.shape)
    world_tensor= world_features_to_one_tensor(lf0,mgc,bap)
    num_padding = mel.shape[0]*8 - world_tensor.shape[0] 
    world_tensor = np.pad(world_tensor, [[0, num_padding], [0, 0]], mode="constant")
    print('features world tensor ',file,':',world_tensor.shape)
    #print('features world2tensor ',file,':',world_tensor.shape,world_tensor_pad.shape)
    lf0,mgc,bap = tensor_to_world_features(world_tensor)
    #print('features tensor2world ',file,':',lf0.shape,mgc.shape,bap.shape)

    wav_s = synthesize(lf0, mgc, bap,sample_rate)
    sf.write('synthesize-'+file, wav_s,sample_rate)
    print('features wav ',file,':',wav_s.shape,wav.shape)
    print('--------------------------------------------------------------')
