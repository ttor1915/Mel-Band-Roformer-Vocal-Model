#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
import numpy as np
import time
import sys
import os
import glob
import torch
import torch.nn as nn
import soundfile as sf
from functools import lru_cache
from tqdm import tqdm
from ml_collections import ConfigDict

# 既存ユーティリティ
from utils import demix_track, get_model_from_config

import warnings
warnings.filterwarnings("ignore")

import torchaudio



# ===== ユーティリティ =====

@lru_cache(maxsize=8)
def _get_resampler(orig_sr: int, target_sr: int, device_str: str):
    """
    (orig_sr, target_sr, device) ごとに Resample をキャッシュ。
    """
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler.to(device_str)


def _load_mono_resampled(path: str, target_sr: int, device: torch.device) -> np.ndarray:
    """
    ogg を torchaudio で読込み、モノラル化し、target_sr にリサンプルして np.float32(1D) を返す。
    """
    # ロード（CPUテンソル [C, T], float32）
    wav_t, sr = torchaudio.load(path)

    # モノラル化（万一ステレオなら 0ch）
    if wav_t.dim() == 2 and wav_t.size(0) > 1:
        wav_t = wav_t[0:1, :]  # [1, T]
    elif wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)  # [1, T]

    if sr != target_sr:
        device_str = str(device) if isinstance(device, torch.device) else "cpu"
        resampler = _get_resampler(sr, target_sr, device_str)  # 既に device_str 上
        wav_t = wav_t.to(device, non_blocking=True)            # 入力だけ device へ
        with torch.inference_mode():
            wav_t = resampler(wav_t)  # [1, T']

    return wav_t.squeeze(0).to("cpu").numpy().astype(np.float32, copy=False)



def _to_mono_numpy(x) -> np.ndarray:
    """
    Tensor/ndarray を 1D float32 モノラル np.ndarray に統一。
    モデル出力が [C, T] または [T] を想定。
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().to("cpu").to(torch.float32).numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # [C, T] 優先（C は通常小さい）
        if arr.shape[0] <= 8:
            return arr[0, :]
        else:
            return arr[:, 0]
    return arr.reshape(-1).astype(np.float32, copy=False)


# ===== メイン処理 =====

def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    # 入力は ogg（モノラル想定）
    all_mixtures_path = sorted(glob.glob(os.path.join(args.input_folder, "*.ogg")))
    total_tracks = len(all_mixtures_path)
    print(f"Total tracks found: {total_tracks}")

    instruments = config.training.instruments
    if getattr(config.training, "target_instrument", None) is not None:
        instruments = [config.training.target_instrument]

    os.makedirs(args.store_dir, exist_ok=True)
    iterator = all_mixtures_path if verbose else tqdm(all_mixtures_path)

    first_chunk_time = None
    target_sr = 16000

    with torch.inference_mode():
        for track_number, path in enumerate(iterator, 1):
            print(f"\nProcessing track {track_number}/{total_tracks}: {os.path.basename(path)}")

            # 1) 読み込み＋16k リサンプル（torchaudio 前提）
            mix = _load_mono_resampled(path, target_sr, device)  # 1D float32 @16k

            # 2) モデル入力整形
            #    ステレオ前提モデルの安全策として 1→2ch 複製（[2, T]）
            mix_stereo = np.stack([mix, mix], axis=0)  # [2, T]
            mixture = torch.from_numpy(mix_stereo).to(torch.float32).to(device, non_blocking=True)

            # 3) 粗い ETA（チャンク推定）
            if first_chunk_time is not None:
                hop = max(1, config.inference.chunk_size // max(1, config.inference.num_overlap))
                total_len = mixture.shape[1]
                num_chunks = (total_len + hop - 1) // hop
                eta = first_chunk_time * num_chunks
                sys.stdout.write(f"Estimated time for this track: ~{eta:.2f} sec\r")
                sys.stdout.flush()

            # 4) 推論
            res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

            # 5) 各パートをモノラル16k FLACで保存
            base = os.path.splitext(os.path.basename(path))[0]
            for instr in instruments:
                out_mono = _to_mono_numpy(res[instr])  # 1D float32
                flac_path = os.path.join(args.store_dir, f"{base}_{instr}_16k.flac")
                sf.write(flac_path, out_mono, target_sr, subtype="PCM_16")

    print(f"\nElapsed time: {time.time() - start_time:.2f} sec")


def proc_folder(cli_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mel_band_roformer")
    parser.add_argument("--config_path", type=str, required=True, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default="", help="Location of the model .pt/.pth")
    parser.add_argument("--input_folder", type=str, required=True, help="folder with mono ogg files")
    parser.add_argument("--store_dir", type=str, required=True, help="output directory")
    parser.add_argument("--device_ids", nargs="+", type=int, default=0, help="GPU id(s) or single int")
    args = parser.parse_args(cli_args) if cli_args is not None else parser.parse_args()

    torch.backends.cudnn.benchmark = True

    with open(args.config_path, "r") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    if args.model_path:
        print(f"Using model: {args.model_path}")
        state = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state)

    if torch.cuda.is_available():
        if isinstance(args.device_ids, int):
            device = torch.device(f"cuda:{args.device_ids}")
            model = model.to(device)
        elif isinstance(args.device_ids, list) and len(args.device_ids) > 1:
            device = torch.device(f"cuda:{args.device_ids[0]}")
            model = nn.DataParallel(model, device_ids=args.device_ids).to(device)
        else:
            device = torch.device(f"cuda:{args.device_ids[0] if isinstance(args.device_ids, list) else args.device_ids}")
            model = model.to(device)
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU (slower).")
        model = model.to(device)

    os.makedirs(args.store_dir, exist_ok=True)
    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
