import argparse
import yaml
import numpy as np
import time
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
from utils import demix_track, get_model_from_config

import warnings
warnings.filterwarnings("ignore")

from pydub import AudioSegment

def _write_mp3_from_float(v, sr, out_path, cbr_bitrate="192k"):
    """
    pydubを使用してfloat型のnumpy配列をMP3に変換して書き出す
    v: np.ndarray, shape=(T,) or (T,2), float32/-1..1
    sr: int
    out_path: str
    """
    # NumPy配列をint16形式に変換 (-1.0～1.0 の範囲を -32767～32767 にスケール)
    int_samples = (v * 32767.0).astype(np.int16)

    # チャンネル数を取得
    channels = v.shape[1] if v.ndim > 1 else 1

    # pydubのAudioSegmentオブジェクトを作成
    audio_segment = AudioSegment(
        int_samples.tobytes(),
        frame_rate=sr,
        sample_width=int_samples.dtype.itemsize,
        channels=channels
    )

    # 元のコードの仕様に合わせてモノラル変換とサンプリングレート変更を行う
    audio_segment = audio_segment.set_channels(1)       # モノラルに変換
    audio_segment = audio_segment.set_frame_rate(16000)   # サンプリングレートを16kHzに変換

    # MP3ファイルとして書き出し
    audio_segment.export(
        out_path,
        format="mp3",
        bitrate=cbr_bitrate,
        parameters=["-compression_level", "0"] # LAMEの高速パス（オプション）
    )



def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + '/*.mp3')
    total_tracks = len(all_mixtures_path)
    print('Total tracks found: {}'.format(total_tracks))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        print(f"\nProcessing track {track_number}/{total_tracks}: {os.path.basename(path)}")

        mix, sr = sf.read(path)
        original_mono = False
        if len(mix.shape) == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        for instr in instruments:
            vocals_output = res[instr].T  # (T, C)
            if original_mono:
                vocals_output = vocals_output[:, 0]  # (T,)

            mp3_path = "{}/{}_{}.mp3".format(
                args.store_dir, os.path.basename(path)[:-4], instr  # ".mp3"→4文字
            )
            _write_mp3_from_float(vocals_output, sr, mp3_path, cbr_bitrate="192k")

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default='', help="Location of the model")
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
    if args.model_path != '':
        print('Using model: {}'.format(args.model_path))
        model.load_state_dict(
            torch.load(args.model_path, map_location=torch.device('cpu'))
        )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids)==int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inference on CPU. It will be very slow...')
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
