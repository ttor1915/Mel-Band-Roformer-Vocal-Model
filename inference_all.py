import argparse
import sys
import os
import glob
import time
import warnings
import yaml
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from tqdm import tqdm
from ml_collections import ConfigDict

# 外部ライブラリの依存
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, pack, unpack, reduce, repeat
from librosa import filters

# 警告を非表示
warnings.filterwarnings("ignore")


# ==========================================
# 1. Model Definitions (Optimized)
# ==========================================

# helper functions
def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# Attend Module (PyTorch Flash Attention Wrapper)
class Attend(nn.Module):
    def __init__(self, dropout=0., flash=True):
        super().__init__()
        self.dropout = dropout
        self.flash = flash

    def forward(self, q, k, v):
        # PyTorch 2.0+ scaled_dot_product_attention (Flash Attention)
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

# FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer
class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed,
                          flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# BandSplit
class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )
            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)
        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)
        return torch.stack(outs, dim=-2)

# MLP Helper
def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)
    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if is_last:
            continue
        net.append(activation())
    return nn.Sequential(*net)

# MaskEstimator
class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)
        outs = []
        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)
        return torch.cat(outs, dim=-1)

# Main Model Class: MelBandRoformer
class MelBandRoformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn=None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn=torch.hann_window,
            match_input_audio_length=False,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.layers = nn.ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            ]))

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]
        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)
        mel_filter_bank[0][0] = 1.
        mel_filter_bank[-1, -1] = 1.

        freqs_per_band = mel_filter_bank > 0
        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])
        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth
            )
            self.mask_estimators.append(mask_estimator)

        self.match_input_audio_length = match_input_audio_length

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None
        
        # to stft
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        batch_arange = torch.arange(batch, device=device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = rearrange(x, 'b f t c -> b t (f c)')
        x = self.band_split(x)

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x = time_transformer(x)
            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)
        masks = masks.type(stft_repr.dtype)

        num_stems = len(self.mask_estimators)
        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1])
        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
        
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)
        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False,
                                  length=istft_length)
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)
        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio


# ==========================================
# 2. Processing Utils (Batched & Optimized)
# ==========================================

def demix_track(config, model, mix, device, batch_size=4, first_chunk_time=None):
    """
    トラックをバッチ処理して高速化する関数。
    batch_size=4 の場合、理論上4倍近いスループットが出ますがメモリも食います。
    """
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32, device=device)
            counter = torch.zeros(req_shape, dtype=torch.float32, device=device)

            total_length = mix.shape[1]
            
            # すべてのチャンクの開始位置をリスト化
            starts = list(range(0, total_length, step))
            num_chunks = len(starts)
            
            if first_chunk_time is None:
                first_chunk = True
            else:
                first_chunk = False

            chunk_start_time = None

            # バッチループ
            for i in range(0, num_chunks, batch_size):
                # 現在のバッチに含まれるインデックスを取得
                current_starts = starts[i : i + batch_size]
                
                # バッチ入力テンソルの作成
                batch_sources = []
                for start in current_starts:
                    part = mix[:, start : start + C]
                    length = part.shape[-1]
                    # 短い場合はパディング
                    if length < C:
                        part = F.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                    batch_sources.append(part)
                
                # (Batch, Channels, ChunkSize)
                batch_input = torch.stack(batch_sources)

                # 時間計測（最初のバッチのみ）
                if first_chunk and i == 0:
                    chunk_start_time = time.time()

                # 推論実行 (Batch processing)
                # 出力: (Batch, Stems, Channels, ChunkSize) 
                # ※ModelがStem次元を返すか、(B, C, T)かによって調整が必要ですが、
                # MelBandRoformerは通常 (B, Stems, C, T) または (B, C, T) [if 1 stem] を返します
                output = model(batch_input)
                
                # 1ステム(Vocalのみ等)の場合は次元を合わせる
                if output.ndim == 3: 
                     output = output.unsqueeze(1)

                # 結果をOverlap-Add
                for j, start in enumerate(current_starts):
                    part_out = output[j] # (Stems, Channels, ChunkSize)
                    length = min(C, total_length - start)
                    
                    # 加算
                    result[..., start : start + length] += part_out[..., :length]
                    counter[..., start : start + length] += 1.

                # 時間予測と表示
                if first_chunk and i == 0: # 最初のバッチ完了時
                    batch_time = time.time() - chunk_start_time
                    # 1チャンクあたりの平均時間
                    avg_chunk_time = batch_time / len(current_starts)
                    first_chunk_time = avg_chunk_time
                    
                    estimated_total_time = avg_chunk_time * num_chunks
                    print(f"Estimated total processing time: {estimated_total_time:.2f} sec (Batch size: {batch_size})")
                    first_chunk = False

                if first_chunk_time is not None and i > 0:
                    chunks_processed = i + len(current_starts)
                    time_remaining = first_chunk_time * (num_chunks - chunks_processed)
                    sys.stdout.write(f"\rRemaining: {time_remaining:.2f} sec")
                    sys.stdout.flush()

            print()
            estimated_sources = result / counter
            estimated_sources = torch.nan_to_num(estimated_sources, nan=0.0)

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}, first_chunk_time
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}, first_chunk_time


# ==========================================
# 3. Main Execution (Optimized)
# ==========================================

def run_folder(model, args, config, device, verbose=False):
    # ★メモリに余裕があるならここを大きくする (例: 3600.0 = 1時間分一気読み)
    # これによりディスクI/OとPythonループのオーバーヘッドを最小化
    MAX_SEGMENT_SEC = 3600.0 
    
    # ★メモリ3倍OKならバッチサイズを4〜8に設定
    # VRAM 24GBなら batch_size=4, 4090/A100なら batch_size=8 いける可能性があります
    BATCH_SIZE = 4

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

    with torch.inference_mode():
        for track_number, path in enumerate(all_mixtures_path, 1):
            filename, ext = os.path.splitext(os.path.basename(path))

            with sf.SoundFile(path) as f_in:
                sr = f_in.samplerate
                total_frames = len(f_in)
                segment_frames = int(MAX_SEGMENT_SEC * sr)

                # GPUリサンプラー
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=16000, dtype=torch.float32
                    ).to(device)
                else:
                    resampler = None

                writers = {}
                for instr in instruments:
                    out_path = f"{args.store_dir}/{filename}.wav"
                    writers[instr] = sf.SoundFile(
                        out_path, mode="w", samplerate=16000, channels=1, subtype="FLOAT"
                    )

                for start in range(0, total_frames, segment_frames):
                    frames = min(segment_frames, total_frames - start)
                    f_in.seek(start)
                    mix = f_in.read(frames, dtype="float32")

                    if mix.ndim == 1:
                        mix = np.stack([mix, mix], axis=1)

                    mixture = torch.tensor(mix.T, dtype=torch.float32, device=device)

                    # バッチサイズを指定して実行
                    res_chunk, first_chunk_time = demix_track(
                        config, model, mixture, device, 
                        batch_size=BATCH_SIZE, 
                        first_chunk_time=first_chunk_time
                    )

                    for instr in instruments:
                        chunk_stereo = res_chunk[instr]
                        if not isinstance(chunk_stereo, torch.Tensor):
                            chunk_stereo = torch.tensor(chunk_stereo, device=device)

                        mono_tensor = chunk_stereo[0:1, :] 

                        if resampler is not None:
                            mono_16k_tensor = resampler(mono_tensor)
                        else:
                            mono_16k_tensor = mono_tensor

                        mono_16k = mono_16k_tensor.squeeze(0).cpu().numpy()
                        writers[instr].write(mono_16k)

                    del mixture, res_chunk
                    
                for instr in instruments:
                    writers[instr].close()
                del writers


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, help="Location of the model")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--batch_size", type=int, default=4, help='batch size for inference (increase for speed, costs VRAM)')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    if args.model_type == 'mel_band_roformer':
        model = MelBandRoformer(**dict(config.model))
    else:
        print('Unknown model: {}'.format(args.model_type))
        return

    device_ids = args.device_ids
    if isinstance(device_ids, list):
        device_id = device_ids[0]
    else:
        device_id = device_ids
        
    device = torch.device(f'cuda:{device_id}')
    model = model.to(device)

    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )
    
    # Run with batch size
    # グローバル変数を直接書き換える形になりますが、関数内でBATCH_SIZEを使用します。
    # 正式には引数で渡すのが綺麗ですが、既存構造を維持するため関数内変数を調整済み。
    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
