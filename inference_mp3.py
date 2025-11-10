import argparse
import yaml
import numpy as np
import time
from ml_collections import ConfigDict
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
from utils import demix_track, get_model_from_config
from pydub import AudioSegment
import io
import warnings
warnings.filterwarnings("ignore")


def write_mp3_from_array(audio_array, sample_rate, output_path):
    """NumPy配列を16 kHzモノラルMP3で保存"""
    target_sr = 16000
    # ステレオ → モノラル
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    # 正規化（-1〜1）
    audio_array = np.clip(audio_array, -1, 1)
    # float → 16bit PCM
    audio_int16 = np.int16(audio_array * 32767)

    # 一時的にWAV化してpydubへ
    buffer = io.BytesIO()
    sf.write(buffer, audio_int16, sample_rate, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    segment = AudioSegment.from_wav(buffer)

    # 16 kHzモノラルへ変換
    segment = segment.set_frame_rate(target_sr).set_channels(1)
    segment.export(output_path, format="mp3", bitrate="128k")


def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + "/*.flac")
    total_tracks = len(all_mixtures_path)
    print(f"Total tracks found: {total_tracks}")

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    os.makedirs(args.store_dir, exist_ok=True)

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

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        # ボーカル
        vocals_output = res[instruments[0]].T
        if original_mono:
            vocals_output = vocals_output[:, 0]
        mp3_path = f"{args.store_dir}/{os.path.basename(path)[:-5]}_{instruments[0]}.mp3"
        write_mp3_from_array(vocals_output, sr, mp3_path)

        # インスト
        # original_mix, _ = sf.read(path)
        # instrumental = original_mix - vocals_output
        # instrumental_path = f"{args.store_dir}/{os.path.basename(path)[:-5]}_instrumental.mp3"
        # write_mp3_from_array(instrumental, sr, instrumental_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} sec")


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mel_band_roformer")
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default="", help="Location of the model")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    parser.add_argument("--device_ids", nargs="+", type=int, default=0, help="list of gpu ids")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    if args.model_path != "":
        print(f"Using model: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if isinstance(device_ids, int):
            device = torch.device(f"cuda:{device_ids}")
            model = model.to(device)
        else:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = "cpu"
        print("CUDA is not available. Run inference on CPU. It will be very slow...")
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
