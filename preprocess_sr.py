import os
import argparse

import torch
import librosa
import json
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile

import utils
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
#import h5py
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import parselmouth
import numpy as np

def get_f0(path,p_len=None, f0_up_key=0):
    x, _ = librosa.load(path, 16000)
    if p_len is None:
        p_len = x.shape[0]//320
    else:
        assert abs(p_len-x.shape[0]//320) < 2, (path, p_len, x.shape)
    time_step = 320 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
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
    basename = os.path.basename(filename)
    print(filename)
    speaker = filename.split("/")[-2]#basename[:4]
    wav_dir = os.path.join(args.wav_dir, speaker)
    ssl_dir = os.path.join(args.ssl_dir, speaker)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(ssl_dir, exist_ok=True)
    wav, _ = librosa.load(filename, sr=hps.sampling_rate)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    mel = mel_spectrogram_torch(
        wav, 
        hps.n_fft, 
        hps.num_mels, 
        hps.sampling_rate, 
        hps.hop_size, 
        hps.win_size, 
        hps.fmin, 
        hps.fmax
    )
    '''
    f = {}
    for i in range(args.min, args.max+1):
        fpath = os.path.join(ssl_dir, f"{i}.hdf5")
        f[i] = h5py.File(fpath, "a")
    '''
    for i in range(68, 92+1, 4):
        mel_rs = utils.transform(mel, i)
        wav_rs = vocoder(mel_rs)[0][0].detach().cpu().numpy()
        _wav_rs = librosa.resample(wav_rs, orig_sr=hps.sampling_rate, target_sr=args.sr)
        wav_rs = torch.from_numpy(_wav_rs).cuda().unsqueeze(0)
        c = utils.get_content(cmodel, wav_rs)
        ssl_path = os.path.join(ssl_dir, basename.replace(".wav", f"_{i}.pt"))
        torch.save(c.cpu(), ssl_path)
        #print(wav_rs.size(), c.size())
        # wav_path = os.path.join(wav_dir, basename.replace(".wav", f"_{i}.wav"))
        # wavfile.write(
        #         wav_path,
        #         args.sr,
        #         _wav_rs
        # )

    len = c.shape[-1]
    cf0, f0 = get_f0(filename, len)
    f0path = filename.replace("22k", "32k")+"f0.npy"
    np.save(f0path, cf0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--min", type=int, default=68, help="min")
    parser.add_argument("--max", type=int, default=92, help="max")
    parser.add_argument("--config", type=str, default="hifigan/config.json", help="path to config file")
    parser.add_argument("--in_dir", type=str, default="dataset/22k", help="path to input dir")
    parser.add_argument("--wav_dir", type=str, default="dataset/sr/wav", help="path to output wav dir")
    parser.add_argument("--ssl_dir", type=str, default="dataset/sr/wavlm", help="path to output ssl dir")
    args = parser.parse_args()

    print("Loading WavLM for content...")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")

    print("Loading vocoder...")
    vocoder = utils.get_vocoder(0)
    vocoder.eval()
    print("Loaded vocoder.")
    
    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)

    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)#[:10]
    
    for filename in tqdm(filenames):
        process(filename)
    