import os
import argparse

import numpy
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging

import parselmouth
import numpy as np

def get_f0(path,p_len=None, f0_up_key=0):
    x, _ = librosa.load(path, 16000)
    if p_len is None:
        p_len = x.shape[0]//160
    else:
        assert abs(p_len-x.shape[0]//160) < 2, (path, p_len, x.shape)
    time_step = 160 / 16000 * 1000
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

logging.getLogger('numba').setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="logs/freevc/G_130000.pth", help="path to pth file")
    parser.add_argument("--outdir", type=str, default="output", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(None)

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    for sample in os.listdir("sample"):
        for i in range(3):
            title = f"{sample[:-4]}-{i}"
            src = f"sample/{sample}"
            tgt = i
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # src
            wav_src, _ = librosa.load(src, sr=16000)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0)
            c = utils.get_content(cmodel, wav_src)
            c = torch.repeat_interleave(c, repeats=2, dim=2)
            print(c.shape)
            g = torch.LongTensor([[tgt]])
            cf0, f0bk = get_f0(src, c.shape[-1])
            f0 = torch.LongTensor(cf0).unsqueeze(0)
            audio = net_g.infer(c,f0=f0, g=g)
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp + "_" + title)), hps.data.sampling_rate,
                      audio)
            else:
                write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)

