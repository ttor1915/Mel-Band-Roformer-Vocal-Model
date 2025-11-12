import argparse
import yaml
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

import numpy as np

import io
import subprocess

def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    # all_mixtures_path = glob.glob(args.input_folder + '/*.wav')
    all_mixtures_path = glob.glob(args.input_folder + '/*')  # WAV以外の拡張子も対象にする
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

        # mix, sr = sf.read(path)
        try:
            mix, sr = sf.read(path)
        except RuntimeError as e:
            print(f"Error reading file {path}: {e}")
            continue  # エラーが発生した場合は次のファイルへ

        # モノラルならステレオに変換（2チャンネル化）
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=1)
            
        mixture = torch.tensor(mix.T, dtype=torch.float32)

        filename, ext = os.path.splitext(os.path.basename(path))

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        # for instr in instruments:
        #     vocals_path = "{}/{}.wav".format(args.store_dir, filename)
        #     sf.write(vocals_path, res[instr].T, sr, subtype='FLOAT')



        for instr in instruments:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, res[instr].T, sr, format='WAV', subtype='FLOAT')
            wav_buffer.seek(0)

            vocals_path = "{}/{}.ogg".format(args.store_dir, filename)


            cmd =  [
                    'ffmpeg',
                    '-y',                    # 出力を上書き
                    '-f', 'wav',             # 入力フォーマット
                    '-i', 'pipe:0',          # 標準入力から読み取る
                    '-acodec', 'libopus',  # 出力コーデック
                    "-ar", "16000",
                    "-ac", "1",
                    "-b:a", "64k",
                    vocals_path             # 出力ファイル
                ]
            # cmd = [
            #         'C:/Users/user/anaconda3/envs/MusicSourceSeparation/Library/bin/ffmpeg.exe',
            #         '-y',                    # 出力を上書き
            #         '-f', 'wav', '-i', 'pipe:0',
            #         '-c:a', 'opus',           # 内蔵opus
            #         '-b:a', '128k',           # 例: ターゲットビットレート
            #         '-compression_level', '10',
            #         '-strict', '-2',          # ← experimental を許可
            #         vocals_path             # 出力ファイル
            #     ]
            subprocess.run(cmd, check=False, capture_output=True, text=True)



    # time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    config_path_defult = '/home/user1/Mel-Band-Roformer-Vocal-Model-main/configs/config_vocals_mel_band_roformer_keytube.yaml'
    start_check_point_defult = '/home/user1/Mel-Band-Roformer-Vocal-Model-main/models/MelBandRoformer.ckpt'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, default=config_path_defult,help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default=start_check_point_defult, help="Location of the model")
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
