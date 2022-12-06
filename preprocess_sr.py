import os
import argparse

import torch
import json
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile

import utils
from mel_processing import mel_spectrogram_torch
#import h5py
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import parselmouth
import librosa
import numpy as np
def stft(y):
    return librosa.stft(
        y=y,
        n_fft=1280,
        hop_length=320,
        win_length=1280,
    )

def energy(y):
    # Extract energy
    S = librosa.magphase(stft(y))[0]
    e = np.sqrt(np.sum(S ** 2, axis=0))  # np.linalg.norm(S, axis=0)
    return e.squeeze()  # (Number of frames) => (654,)

def get_energy(path, p_len=None):
    wav, sr = librosa.load(path, 48000)
    e = energy(wav)
    if p_len is None:
        p_len = wav.shape[0] // 320
    assert e.shape[0] -p_len <2 ,(e.shape[0] ,p_len)
    e = e[: p_len]
    return e



def get_f0(path,p_len=None, f0_up_key=0):
    x, _ = librosa.load(path, 48000)
    if p_len is None:
        p_len = x.shape[0]//320
    else:
        assert abs(p_len-x.shape[0]//320) < 3, (path, p_len, x.shape)
    time_step = 320 / 48000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 48000).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    f0bak = f0.copy()
    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0bak

def process(filename):
    print(filename)
    devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, _ = librosa.load(filename, sr=args.sr)
    wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
    c = utils.get_hubert_content(hmodel, wav)
    save_name = filename.replace("16k", "48k")+".soft.pt"
    torch.save(c.cpu(), save_name)

    cf0, f0 = get_f0(filename.replace("16k", "48k"), c.shape[-1] * 3)
    f0path = filename.replace("16k", "48k")+".f0.npy"
    np.save(f0path, f0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="dataset/16k", help="path to input dir")
    args = parser.parse_args()

    print("Loading hubert for content...")
    hmodel = utils.get_hubert_model(0 if torch.cuda.is_available() else None)
    print("Loaded hubert.")

    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)#[:10]
    
    for filename in tqdm(filenames):
        process(filename)
    