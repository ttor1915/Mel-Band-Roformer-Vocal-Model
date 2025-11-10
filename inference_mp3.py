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

# オプション: torchaudio があれば高速リサンプルに使用
try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False

# librosa はフォールバック用に残す
import librosa


# 先頭付近（importの下）に追加
from functools import lru_cache

@lru_cache(maxsize=8)
def _get_resampler(orig_sr: int, target_sr: int, device_str: str):
    """
    同一 (orig_sr, target_sr, device) の Resample モジュールをキャッシュ。
    torchaudio が無い環境では None を返す。
    """
    try:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        # CPU/GPU どちらでも動作。GPU のときは to(device) で転送。
        return resampler.to(device_str)
    except Exception:
        return None


def _fast_resample_mono(wave_np: np.ndarray, orig_sr: int, target_sr: int, device: torch.device):
    """
    モノラル1D ndarray -> 1D ndarray（target_srへリサンプル）
    可能なら torchaudio.transforms.Resample を使用（高速）。
    無ければ librosa.resample(..., res_type="kaiser_fast") にフォールバック。
    """
    if orig_sr == target_sr:
        return wave_np, orig_sr

    device_str = str(device) if isinstance(device, torch.device) else "cpu"

    # 1) torchaudio で高速リサンプル
    try:
        import torchaudio  # 動的 import
        wav_t = torch.from_numpy(wave_np).to(torch.float32).unsqueeze(0)  # [1, T]
        # 入力テンソルも resampler も同じ device へ
        wav_t = wav_t.to(device if torch.cuda.is_available() else "cpu", non_blocking=True)

        resampler = _get_resampler(orig_sr, target_sr, device_str)
        if resampler is not None:
            with torch.no_grad():
                res_t = resampler(wav_t)  # 引数名は不要（位置引数でも可）
            res = res_t.squeeze(0).to("cpu").numpy()
            return res, target_sr
    except Exception:
        pass

    # 2) フォールバック（十分高速・高品質）
    print('librosa使用')
    res = librosa.resample(wave_np, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_fast")
    return res.astype(np.float32, copy=False), target_sr

def _to_mono_numpy(x) -> np.ndarray:
    """
    Tensor/ndarray を受け取り、モノラル 1D の np.float32 にして返す。
    [C, T] の場合は 0ch を採用。 [T, C] が来た場合は 0列を採用。
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().to("cpu").to(torch.float32).numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1:
        return arr  # [T]

    if arr.ndim == 2:
        # まず [C, T] を優先。そうでなければ [T, C] とみなす。
        if arr.shape[0] <= 8:  # C が小さい想定
            # [C, T] -> 0ch
            return arr[0, :]
        else:
            # [T, C] -> 0列
            return arr[:, 0]

    # それ以外はフラット化（安全側）
    return arr.reshape(-1)



def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    # 入力はモノラルMP3のみを仮定
    all_mixtures_path = sorted(glob.glob(os.path.join(args.input_folder, "*.mp3")))
    total_tracks = len(all_mixtures_path)
    print(f"Total tracks found: {total_tracks}")

    instruments = config.training.instruments
    if getattr(config.training, "target_instrument", None) is not None:
        instruments = [config.training.target_instrument]

    os.makedirs(args.store_dir, exist_ok=True)

    paths_iter = all_mixtures_path if verbose else tqdm(all_mixtures_path)

    first_chunk_time = None
    target_sr = 16000  # 固定出力

    # 以降は入力モノラル前提の簡潔パス
    with torch.inference_mode():
        for track_number, path in enumerate(paths_iter, 1):
            print(f"\nProcessing track {track_number}/{total_tracks}: {os.path.basename(path)}")

            # 1) 音声読込（モノラル前提: 1D shape）
            #    dtype=float32 で固定、always_2d=False で 1D を受け取る
            mix, sr = sf.read(path, dtype="float32", always_2d=False)

            # 念のためモノラル確認（想定外入力の際は 1ch抽出）
            if mix.ndim > 1:
                mix = mix[:, 0].astype(np.float32, copy=False)

            # 2) 16kHzへリサンプル（高速経路を優先）
            if sr != target_sr:
                mix, sr = _fast_resample_mono(mix, sr, target_sr, device)

            # 3) モデル入力テンソル整形
            #    モデルがステレオ前提の可能性を考慮し 1ch→2ch複製（コスト僅少）
            #    ※モデルが1ch対応なら channels=1 にしてもOK
            #    下行を channels=1 に変えて試す場合:
            #        mixture = torch.from_numpy(mix[None, :])  # [1, T]
            mix_stereo = np.stack([mix, mix], axis=0)  # [2, T]
            mixture = torch.from_numpy(mix_stereo)     # float32, [C, T]
            mixture = mixture.to(device, non_blocking=True)

            # 4) 1チャンク目の時間から粗いETAを表示（I/Oや保存時間は含まず）
            if first_chunk_time is not None:
                total_length = mixture.shape[1]
                hop = max(1, config.inference.chunk_size // config.inference.num_overlap)
                num_chunks = (total_length + hop - 1) // hop
                estimated_total_time = first_chunk_time * num_chunks
                sys.stdout.write(f"Estimated time for this track: ~{estimated_total_time:.2f} sec\r")
                sys.stdout.flush()

            # 5) 推論
            res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

            # 6) 出力（各パートをモノラルで16k FLAC保存）
            for instr in instruments:
                out = res[instr]
                out_mono = _to_mono_numpy(out)

                # パス
                base = os.path.splitext(os.path.basename(path))[0]
                out_path = f"{args.store_dir}/{base}_{instr}_16k.flac"

                # FLAC保存（16-bit PCM）
                sf.write(out_path, out_mono, sr, subtype="PCM_16")

    print(f"\nElapsed time: {time.time() - start_time:.2f} sec")


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

    # cuDNN 最適化
    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    if args.model_path != "":
        print(f"Using model: {args.model_path}")
        state = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state)

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if isinstance(device_ids, int):
            device = torch.device(f"cuda:{device_ids}")
            model = model.to(device)
        else:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Run inference on CPU (slower).")
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
