import argparse
import yaml
# import time
from ml_collections import ConfigDict
# from omegaconf import OmegaConf
from tqdm import tqdm
# import sys
import os
import glob
import torch
import soundfile as sf
# import torch.nn as nn
from utils import demix_track, get_model_from_config

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import librosa


def run_folder(model, args, config, device, verbose=False):
    # 長すぎる音声を避けるための最大チャンク長（秒）
    # 必要に応じて 60.0 を 30.0 や 120.0 に変更してください
    MAX_SEGMENT_SEC = 1200.0

    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + '/*.ogg')

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    # チャンク処理時間を demix_track に渡すための変数
    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        filename, ext = os.path.splitext(os.path.basename(path))

        # 音声を一括で読み込まず、SoundFile でチャンク単位に読み込む
        with sf.SoundFile(path) as f:
            sr = f.samplerate
            total_frames = len(f)
            segment_frames = int(MAX_SEGMENT_SEC * sr)

            # 楽器ごとにチャンク結果を溜めるバッファ
            res_segments = {instr: [] for instr in instruments}

            # ファイル全体を segment_frames ごとに分割して処理
            for start in range(0, total_frames, segment_frames):
                f.seek(start)
                frames = min(segment_frames, total_frames - start)

                # mix: shape = (frames, channels) or (frames,)
                mix = f.read(frames, dtype='float32')

                # モノラルならステレオに変換（2チャンネル化）
                if mix.ndim == 1:
                    mix = np.stack([mix, mix], axis=1)  # (frames, 2)

                # demix_track が期待する形 (channels, samples) に転置
                mixture = torch.tensor(mix.T, dtype=torch.float32)

                # チャンクごとに分離
                res_chunk, first_chunk_time = demix_track(
                    config, model, mixture, device, first_chunk_time
                )

                # 楽器ごとに結果を蓄積（時間方向に後で結合）
                for instr in instruments:
                    res_segments[instr].append(res_chunk[instr])

            # 全チャンクを時間方向（サンプル方向）に連結
            res = {}
            for instr in instruments:
                # res_chunk[instr] は shape = (channels, samples) を想定
                res[instr] = np.concatenate(res_segments[instr], axis=1)

        # ここからは従来どおり：モノラル化 → 16kHz リサンプリング → 保存
        for instr in instruments:
            # 元の波形（res[instr]）をモノラル化
            mono = librosa.to_mono(res[instr])

            # 16 kHz にリサンプリング
            mono_16k = librosa.resample(mono, orig_sr=sr, target_sr=16000)

            vocals_path = f"{args.store_dir}/{filename}.wav"

            # 16k・モノラルで保存
            sf.write(vocals_path, mono_16k, 16000, subtype='FLOAT')



def proc_folder(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str,help="path to config yaml file")
    parser.add_argument("--model_path", type=str, help="Location of the model")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
      config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)

    device_ids = args.device_ids
    device = torch.device(f'cuda:{device_ids}')
    model = model.to(device)

    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
