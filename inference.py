import argparse
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm
import os
import glob
import torch
import soundfile as sf
from utils import demix_track, get_model_from_config

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa


def run_folder(model, args, config, device, verbose=False):
    # 1チャンクの長さ（秒）。GPUメモリに合わせて調整。
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

    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        filename, ext = os.path.splitext(os.path.basename(path))

        # 入力ファイルをストリーミング読み込み
        with sf.SoundFile(path) as f_in:
            sr = f_in.samplerate
            total_frames = len(f_in)
            segment_frames = int(MAX_SEGMENT_SEC * sr)

            # 出力ファイルを楽器ごとに開く（16k, モノラル）
            writers = {}
            for instr in instruments:
                # 必要に応じて f"{filename}.wav" に戻してもよい
                out_path = f"{args.store_dir}/{filename}_{instr}.wav"
                writers[instr] = sf.SoundFile(
                    out_path,
                    mode="w",
                    samplerate=16000,
                    channels=1,
                    subtype="FLOAT",
                )

            # ファイル全体をチャンクに分けて処理
            for start in range(0, total_frames, segment_frames):
                frames = min(segment_frames, total_frames - start)
                f_in.seek(start)
                mix = f_in.read(frames, dtype="float32")  # (frames,) or (frames, ch)

                # モノラルならステレオに複製
                if mix.ndim == 1:
                    mix = np.stack([mix, mix], axis=1)  # (frames, 2)

                # (channels, samples) に転置してテンソル化
                mixture = torch.tensor(mix.T, dtype=torch.float32)

                # demix（チャンク単位）
                res_chunk, first_chunk_time = demix_track(
                    config, model, mixture, device, first_chunk_time
                )

                # メモリを抑えるため、チャンクごとにモノラル＋16kにして即書き出し
                for instr in instruments:
                    chunk_stereo = res_chunk[instr]

                    # torch.Tensor の場合は numpy に変換
                    if isinstance(chunk_stereo, torch.Tensor):
                        chunk_stereo = chunk_stereo.detach().cpu().numpy()

                    # chunk_stereo: (channels, samples) を想定
                    # モノラル化（librosa.to_mono でもよいが平均を直接取る方が軽い）
                    mono = chunk_stereo.mean(axis=0).astype(np.float32)  # (samples,)

                    # 16kHz にリサンプリング
                    mono_16k = librosa.resample(
                        mono, orig_sr=sr, target_sr=16000
                    ).astype(np.float32)

                    # 即ファイルに追記
                    writers[instr].write(mono_16k)

                    # チャンク内の一時配列を解放しやすくする
                    del chunk_stereo, mono, mono_16k

                # 入力チャンクも解放
                del mix, mixture, res_chunk

            # 出力ファイルを閉じる
            for instr in instruments:
                writers[instr].close()
            del writers



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
