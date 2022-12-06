import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm


def process(item):
    spkdir, wav_name, args = item
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = spkdir.split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir1, speaker), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, None)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        save_name = wav_name
        save_path1 = os.path.join(args.out_dir1, speaker, save_name)
        save_path2 = os.path.join(args.out_dir2, speaker, save_name)
        wavfile.write(
            save_path1,
            args.sr1,
            (wav1 * np.iinfo(np.int16).max).astype(np.int16)
        )
        wavfile.write(
            save_path2,
            args.sr2,
            (wav2 * np.iinfo(np.int16).max).astype(np.int16)
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--sr2", type=int, default=48000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="/home/Datasets/lijingyi/data/vctk/wav48_silence_trimmed/", help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default="./dataset/16k", help="path to target dir")
    parser.add_argument("--out_dir2", type=str, default="./dataset/48k", help="path to target dir")
    args = parser.parse_args()
    processs = cpu_count()-2 if cpu_count() >4 else 1
    pool = Pool(processes=processs)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            print([(spk_dir, i) for i in os.listdir(spk_dir) if i.endswith("wav")])
            for _ in tqdm(pool.imap_unordered(process, [(spk_dir, i, args) for i in os.listdir(spk_dir) if i.endswith("wav")])):
                pass
