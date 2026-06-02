"""
Streaming MaxSim retriever for SimulEval inference.

Replaces the FAISS-based sliding-window retriever with direct MaxSim scoring,
matching the retrieval mechanism used during Speech LLM training data generation.

The audio chunk (whatever length) is passed directly to the MaxSim model, which
handles multi-scale windowing internally via _multiscale_pool.

Reuses model definitions from qwen3_glossary_neg_train.py:
  - Qwen3OmniRetriever (use_maxsim=True)
  - BgeM3TextEncoder
  - _maxsim_score
"""

from __future__ import annotations

# ======Configuration=====
AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
RAG_FEATURE_EXTRACTOR_MODEL_ID = "openai/whisper-large-v3"
EXPECTED_SAMPLE_RATE = 16000
ENCODER_FPS = 12.5
FRAME_SEC = 1.0 / ENCODER_FPS

LORA_RANK = 128
LORA_ALPHA = 256
POOLING_TYPE = "transformer"
TEMPERATURE = 0.03
USE_MAXSIM = True
# Match the actual retriever checkpoint used for TCM final v3 training.
# The training script default is older; the launched run overrode it explicitly.
MAXSIM_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
MAXSIM_STRIDE = 2
TARGET_DIM = 1024

TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7

LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()
# ======Configuration=====

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(
    os.environ.get("RASST_ACTIVE_CODE_ROOT", Path(__file__).resolve().parents[2])
)


def _import_model_classes():
    """Import model classes from the training code."""
    train_dir = _REPO_ROOT / "retriever"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    from qwen3_glossary_neg_train import (
        BgeM3TextEncoder,
        Qwen3OmniRetriever,
        _maxsim_score,
    )
    return Qwen3OmniRetriever, BgeM3TextEncoder, _maxsim_score


def _build_retriever_model(
    device: torch.device,
    lora_rank: int,
    text_lora_rank: int,
    maxsim_windows: Optional[List[int]] = None,
    maxsim_stride: int = MAXSIM_STRIDE,
):
    """Build MaxSim retriever and text encoder with given LoRA ranks."""
    Qwen3OmniRetriever, BgeM3TextEncoder, _maxsim_score = _import_model_classes()
    from transformers import WhisperFeatureExtractor

    retriever = Qwen3OmniRetriever(
        model_id=AUDIO_MODEL_ID,
        target_dim=TARGET_DIM,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_target_modules=LORA_TARGET_MODULES,
        temperature=TEMPERATURE,
        learn_temp=False,
        pooling_type=POOLING_TYPE,
        use_maxsim=USE_MAXSIM,
        maxsim_windows=maxsim_windows or MAXSIM_WINDOWS,
        maxsim_stride=maxsim_stride,
    ).to(device)

    feat_ext = WhisperFeatureExtractor.from_pretrained(RAG_FEATURE_EXTRACTOR_MODEL_ID)

    return retriever, feat_ext, _maxsim_score


def _load_checkpoint(
    retriever, model_path: str, device: torch.device
) -> None:
    """Load checkpoint weights into audio retriever (text encoder already pre-encoded in index)."""
    ckpt = torch.load(model_path, map_location=device)

    def _strip(sd):
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }

    retriever.load_state_dict(
        _strip(ckpt.get("model_state_dict", {})), strict=False
    )
    retriever.eval()
    logger.info("Loaded MaxSim retriever checkpoint from %s", model_path)


@torch.no_grad()
def _encode_audio(
    audio_array: np.ndarray,
    retriever,
    feat_ext,
    device: torch.device,
) -> torch.Tensor:
    """Encode a single variable-length audio array into MaxSim multi-window embeddings.

    Returns: [1, W, D] tensor of L2-normalized window embeddings.
    """
    inp = feat_ext(
        [audio_array],
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    mel = inp.input_features.squeeze(0)  # [C, T_mel]
    mel_len = mel.shape[-1]
    input_features = mel.to(device, dtype=torch.bfloat16)
    feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embs = retriever(input_features, feature_lens)
    return embs.float()  # [1, W, D]


def _compute_sim(
    speech_emb: torch.Tensor,
    text_embs: torch.Tensor,
    maxsim_score_fn,
) -> torch.Tensor:
    """Compute raw cos-sim / max-sim logits [1, G] for one audio chunk."""
    if speech_emb.ndim == 3:
        return maxsim_score_fn(speech_emb, text_embs)
    return speech_emb @ text_embs.T


def _build_window_time_ranges(
    maxsim_windows: List[int], maxsim_stride: int, t_frames: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build window start/end times in seconds, matching _multiscale_pool order."""
    starts = []
    ends = []
    for w in maxsim_windows:
        if w >= t_frames:
            starts.append(0.0)
            ends.append(t_frames * FRAME_SEC)
        else:
            n_out = (t_frames - w) // maxsim_stride + 1
            for p in range(n_out):
                frame_start = p * maxsim_stride
                frame_end = frame_start + w
                starts.append(frame_start * FRAME_SEC)
                ends.append(frame_end * FRAME_SEC)
    return (
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor(ends, dtype=torch.float32),
    )


@torch.no_grad()
def _encode_audio_projected_seq(
    audio_array: np.ndarray,
    retriever,
    feat_ext,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode audio to projected encoder frames before MaxSim pooling.

    This mirrors Qwen3OmniRetriever.forward up to `projected_seq`, so inference
    can pool over a continuous previous+current frame sequence and attach
    per-window timeline metadata.
    """
    inp = feat_ext(
        [audio_array],
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    mel = inp.input_features.squeeze(0)  # [C, T_mel]
    mel_len = mel.shape[-1]
    input_features = mel.to(device, dtype=torch.bfloat16)
    feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = retriever.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state

        if hidden_states.ndim == 2:
            output_lens: List[int] = []
            for cur in feature_lens.tolist():
                reduced = cur
                for _ in range(3):
                    reduced = (reduced + 1) // 2
                output_lens.append(reduced)
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(x / ratio)) for x in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            from torch.nn.utils.rnn import pad_sequence

            parts = torch.split(hidden_states, output_lens, dim=0)
            hidden_states = pad_sequence(parts, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)

        batch_size, max_len, _ = hidden_states.shape
        mask = (
            torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len)
            < feature_lens.unsqueeze(1)
        )
        projected_seq = retriever.projector(hidden_states)
        projected_seq = projected_seq * mask.unsqueeze(-1).float()

    return projected_seq.float(), mask


def _audio_output_lens_tensor(feature_lens: torch.Tensor) -> torch.Tensor:
    """Infer Qwen3-Omni AuT output lengths as a tensor."""
    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            _get_feat_extract_output_lengths,
        )

        return _get_feat_extract_output_lengths(feature_lens).clamp(min=1).long()
    except Exception:
        lens = feature_lens.long()
        for _ in range(3):
            lens = (lens + 1) // 2
        return lens.clamp(min=1)


def _audio_output_lens(
    feature_lens: torch.Tensor,
    *,
    total_hidden_len: int,
    total_input_len: int,
) -> List[int]:
    """Infer Qwen3-Omni AuT output lengths for packed audio samples."""
    output_lens = [
        max(1, int(value))
        for value in _audio_output_lens_tensor(feature_lens).tolist()
    ]

    if sum(output_lens) != int(total_hidden_len):
        output_lens = []
        for cur in feature_lens.tolist():
            reduced = int(cur)
            for _ in range(3):
                reduced = (reduced + 1) // 2
            output_lens.append(max(1, int(reduced)))

    if sum(output_lens) != int(total_hidden_len):
        ratio = float(total_input_len) / max(float(total_hidden_len), 1.0)
        output_lens = [
            max(1, round(float(cur) / ratio))
            for cur in feature_lens.tolist()
        ]

    if output_lens:
        output_lens[-1] = int(total_hidden_len) - sum(output_lens[:-1])
        output_lens[-1] = max(1, output_lens[-1])
    return output_lens


def _audio_encoder_forward_packed_isolated(
    audio_encoder,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3-Omni AuT packed forward with sample/chunk isolation.

    HF's AuT forward relies on FlashAttention varlen kernels for true packed
    isolation. In eager/SDPA environments, ``cu_seqlens`` is otherwise ignored
    unless an explicit block attention mask is passed to each encoder layer.
    """
    aftercnn_lens = _audio_output_lens_tensor(feature_lens)
    chunk_num = torch.ceil(feature_lens / (audio_encoder.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [audio_encoder.n_window * 2] * int(chunk_num.sum().item()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (audio_encoder.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = audio_encoder.n_window * 2

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _audio_output_lens_tensor(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [
            torch.ones(int(length), dtype=torch.bool, device=padded_feature.device)
            for length in feature_lens_after_cnn
        ],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)

    padded_embeds = []
    for chunk in padded_feature.split(audio_encoder.conv_chunksize, dim=0):
        padded_embed = F.gelu(audio_encoder.conv2d1(chunk))
        padded_embed = F.gelu(audio_encoder.conv2d2(padded_embed))
        padded_embed = F.gelu(audio_encoder.conv2d3(padded_embed))
        padded_embeds.append(padded_embed)
    padded_embed = torch.cat(padded_embeds, dim=0)
    batch_chunks, channels, freq, frames = padded_embed.size()
    padded_embed = audio_encoder.conv_out(
        padded_embed.permute(0, 3, 1, 2)
        .contiguous()
        .view(batch_chunks, frames, channels * freq)
    )

    positional_embedding = (
        audio_encoder.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]

    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (
        audio_encoder.n_window_infer // (audio_encoder.n_window * 2)
    )
    for cnn_len in aftercnn_lens.tolist():
        cu_chunk_lens += [window_aftercnn] * (int(cnn_len) // window_aftercnn)
        remainder = int(cnn_len) % window_aftercnn
        if remainder != 0:
            cu_chunk_lens.append(remainder)
    cu_seqlens = torch.tensor(
        cu_chunk_lens,
        device=feature_lens.device,
    ).cumsum(-1, dtype=torch.int32)

    attention_mask = None
    if getattr(audio_encoder.config, "_attn_implementation", "eager") != "flash_attention_2":
        attention_mask = audio_encoder._prepare_attention_mask(hidden_states, cu_seqlens)

    for encoder_layer in audio_encoder.layers:
        layer_outputs = encoder_layer(
            hidden_states,
            cu_seqlens,
            attention_mask=attention_mask,
        )
        hidden_states = layer_outputs[0]

    hidden_states = audio_encoder.ln_post(hidden_states)
    hidden_states = audio_encoder.proj1(hidden_states)
    hidden_states = audio_encoder.act(hidden_states)
    hidden_states = audio_encoder.proj2(hidden_states)
    return hidden_states


def _project_hidden_states(
    hidden_states: torch.Tensor,
    feature_lens: torch.Tensor,
    retriever,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_len, _ = hidden_states.shape
    mask = (
        torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len)
        < feature_lens.unsqueeze(1)
    )
    projected_seq = retriever.projector(hidden_states)
    projected_seq = projected_seq * mask.unsqueeze(-1).float()
    return projected_seq.float(), mask


def _feature_extract_mels(
    audio_arrays: Sequence[np.ndarray],
    feat_ext,
) -> List[torch.Tensor]:
    audio_lens = [int(np.asarray(audio).shape[0]) for audio in audio_arrays]
    if any(length <= 0 for length in audio_lens):
        raise ValueError("audio_arrays contains an empty audio segment")

    if len(set(audio_lens)) == 1:
        inp = feat_ext(
            list(audio_arrays),
            sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        return [inp.input_features[i] for i in range(len(audio_arrays))]

    mel_list: List[torch.Tensor] = []
    for audio in audio_arrays:
        inp = feat_ext(
            [audio],
            sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        mel_list.append(inp.input_features.squeeze(0))  # [C, T_mel_i]
    return mel_list


@torch.no_grad()
def _encode_audio_projected_seq_batch_serial(
    mel_list: Sequence[torch.Tensor],
    retriever,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a mel batch by running the AuT forward path per sample."""
    projected_parts: List[torch.Tensor] = []
    mask_parts: List[torch.Tensor] = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for mel in mel_list:
            mel_len = int(mel.shape[-1])
            input_features = mel.to(device, dtype=torch.bfloat16)
            feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)
            outputs = retriever.audio_encoder(input_features, feature_lens)
            hidden_states = outputs.last_hidden_state

            if hidden_states.ndim == 2:
                output_lens = _audio_output_lens(
                    feature_lens,
                    total_hidden_len=int(hidden_states.shape[0]),
                    total_input_len=int(input_features.shape[1]),
                )
                hidden_states = hidden_states.unsqueeze(0)
                feature_lens = torch.tensor(output_lens, device=hidden_states.device)

            projected_seq, mask = _project_hidden_states(
                hidden_states,
                feature_lens,
                retriever,
            )
            projected_parts.append(projected_seq.squeeze(0).float())
            mask_parts.append(mask.squeeze(0))

    from torch.nn.utils.rnn import pad_sequence

    projected_seq = pad_sequence(projected_parts, batch_first=True)
    mask = pad_sequence(mask_parts, batch_first=True, padding_value=False)
    return projected_seq.float(), mask


@torch.no_grad()
def _encode_audio_projected_seq_batch_packed(
    mel_list: Sequence[torch.Tensor],
    retriever,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a mel batch using Qwen3-Omni AuT's packed sequence contract.

    Qwen3-Omni's HF/SGLang audio tower expects a packed mel stream
    ``[C, sum(T_i)]`` plus per-sample ``feature_lens``. The tower then builds
    chunk-level ``cu_seqlens`` internally so attention does not cross sample
    boundaries.
    """
    if not mel_list:
        raise ValueError("mel_list must be non-empty")

    mel_lens = [int(mel.shape[-1]) for mel in mel_list]
    input_features = torch.cat(
        [mel.to(device, dtype=torch.bfloat16) for mel in mel_list],
        dim=-1,
    )
    feature_lens = torch.tensor(mel_lens, dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if os.environ.get("RASST_RAG_AUT_BATCH_ISOLATE", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            hidden_states = _audio_encoder_forward_packed_isolated(
                retriever.audio_encoder,
                input_features,
                feature_lens,
            )
        else:
            outputs = retriever.audio_encoder(input_features, feature_lens)
            hidden_states = outputs.last_hidden_state

        if hidden_states.ndim == 2:
            output_lens = _audio_output_lens(
                feature_lens,
                total_hidden_len=int(hidden_states.shape[0]),
                total_input_len=int(input_features.shape[1]),
            )
            from torch.nn.utils.rnn import pad_sequence

            parts = torch.split(hidden_states, output_lens, dim=0)
            hidden_states = pad_sequence(parts, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)

        projected_seq, mask = _project_hidden_states(
            hidden_states,
            feature_lens,
            retriever,
        )

    return projected_seq.float(), mask


@torch.no_grad()
def _encode_audio_projected_seq_batch(
    audio_arrays: Sequence[np.ndarray],
    retriever,
    feat_ext,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode variable-length audio to projected encoder frames.

    By default this uses the packed AuT path expected by Qwen3-Omni:
    ``[C, sum(T_i)]`` plus ``feature_lens``. Set
    ``RASST_RAG_AUT_BATCH_MODE=serial`` to force the per-sample path, or
    ``packed_strict`` to disable fallback on packed-forward errors.
    """
    if not audio_arrays:
        raise ValueError("_encode_audio_projected_seq_batch requires non-empty audio_arrays")

    mel_list = _feature_extract_mels(audio_arrays, feat_ext)
    mode = os.environ.get("RASST_RAG_AUT_BATCH_MODE", "packed").strip().lower()
    if mode in {"serial", "per_sample", "per-sample", "0", "false"}:
        return _encode_audio_projected_seq_batch_serial(mel_list, retriever, device)

    try:
        return _encode_audio_projected_seq_batch_packed(mel_list, retriever, device)
    except Exception as exc:
        if mode in {"packed_strict", "strict"}:
            raise
        logger.warning(
            "Packed Qwen3-Omni AuT batch failed; falling back to serial mode: %s",
            exc,
        )
        return _encode_audio_projected_seq_batch_serial(mel_list, retriever, device)


def _retrieve_topk(
    speech_emb: torch.Tensor,
    text_embs: torch.Tensor,
    maxsim_score_fn,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k glossary terms for one audio chunk."""
    sim = _compute_sim(speech_emb, text_embs, maxsim_score_fn)  # [1, G]
    scores = sim.squeeze(0)
    n = min(k, scores.numel())
    top_sco, top_idx = torch.topk(scores, k=n, largest=True, sorted=True)
    return top_idx.cpu().numpy(), top_sco.cpu().numpy()


class StreamingMaxSimRetriever:
    """MaxSim-based retriever for streaming simultaneous translation.

    Replaces FAISS-based retrieval with direct MaxSim scoring against
    pre-encoded text embeddings from a .pt index file.
    """

    def __init__(
        self,
        model_path: str,
        index_path: str,
        device: str = "cuda:1",
        top_k: int = 10,
        lora_rank: int = LORA_RANK,
        text_lora_rank: int = TEXT_LORA_RANK,
        target_lang: str = "zh",
        window_sec: float = 0.0,
        score_threshold: float = 0.0,
        maxsim_windows: Optional[List[int]] = None,
        maxsim_stride: int = MAXSIM_STRIDE,
    ):
        self.device = torch.device(device)
        self.top_k = top_k
        self.target_lang = target_lang
        self.window_sec = window_sec
        self.score_threshold = float(score_threshold)
        self.enabled = False

        assert model_path and Path(model_path).is_file(), (
            f"MaxSim model checkpoint not found: {model_path}"
        )
        assert index_path and Path(index_path).is_file(), (
            f"MaxSim text index not found: {index_path}"
        )

        logger.info(
            "Loading MaxSim retriever: model=%s index=%s device=%s lora_r=%d "
            "window_sec=%.2f maxsim_windows=%s maxsim_stride=%d",
            model_path,
            index_path,
            device,
            lora_rank,
            window_sec,
            maxsim_windows or MAXSIM_WINDOWS,
            maxsim_stride,
        )

        self.retriever, self.feat_ext, self._maxsim_score = _build_retriever_model(
            self.device,
            lora_rank,
            text_lora_rank,
            maxsim_windows=maxsim_windows,
            maxsim_stride=maxsim_stride,
        )
        _load_checkpoint(self.retriever, model_path, self.device)

        index_data = torch.load(index_path, map_location=self.device)
        self.text_embs = index_data["text_embs"].to(self.device)
        self.term_list = index_data["term_list"]
        assert self.text_embs.shape[0] == len(self.term_list), (
            f"Index mismatch: text_embs has {self.text_embs.shape[0]} entries "
            f"but term_list has {len(self.term_list)}"
        )

        self._audio_buffer: Optional[np.ndarray] = None
        self._retrieve_offset: int = 0
        self._last_results: List[Dict] = []

        self.enabled = True
        logger.info(
            "MaxSim retriever ready: %d glossary terms, text_embs shape=%s",
            len(self.term_list), list(self.text_embs.shape),
        )

    def reset(self) -> None:
        """Reset state for a new utterance."""
        self._audio_buffer = None
        self._retrieve_offset = 0
        self._last_results = []

    def get_audio_duration(self) -> float:
        """Return duration of accumulated audio in seconds."""
        if self._audio_buffer is None:
            return 0.0
        return float(len(self._audio_buffer)) / EXPECTED_SAMPLE_RATE

    def accumulate_audio(self, new_audio: Optional[np.ndarray]) -> None:
        """Append new audio samples to the internal buffer."""
        if new_audio is None or len(new_audio) == 0:
            return
        audio = np.asarray(new_audio, dtype=np.float32).flatten()
        if self._audio_buffer is None:
            self._audio_buffer = audio
        else:
            self._audio_buffer = np.concatenate([self._audio_buffer, audio])

    def retrieve(self, top_k: Optional[int] = None) -> List[Dict]:
        """Run MaxSim retrieval using a sliding window over accumulated audio.

        When ``window_sec > 0``, encodes the most recent ``window_sec`` of audio
        (reaching back into previously processed regions).  This gives 50%
        overlap between consecutive calls when the caller invokes retrieve()
        every ``window_sec / 2`` seconds.

        When ``window_sec <= 0`` (legacy mode), encodes only audio since the
        last retrieve() call (no overlap).

        Returns list of dicts with keys: term, translation, key, score.
        """
        if self._audio_buffer is None or len(self._audio_buffer) == 0:
            return self._last_results

        buffer_end = len(self._audio_buffer)
        new_since_last = buffer_end - self._retrieve_offset
        min_samples = int(0.48 * EXPECTED_SAMPLE_RATE)
        if new_since_last < min_samples:
            return self._last_results

        if self.window_sec > 0:
            window_samples = int(self.window_sec * EXPECTED_SAMPLE_RATE)
            start = max(0, buffer_end - window_samples)
        else:
            start = self._retrieve_offset

        chunk = self._audio_buffer[start:buffer_end]

        k = top_k if top_k is not None else self.top_k

        embs = _encode_audio(
            chunk, self.retriever, self.feat_ext, self.device
        )

        idx_arr, sco_arr = _retrieve_topk(
            embs, self.text_embs, self._maxsim_score, k
        )

        self._retrieve_offset = buffer_end

        results = self._build_results(idx_arr, sco_arr)
        self._last_results = results
        return results

    def retrieve_timeline(
        self,
        top_k: Optional[int],
        current_start_sec: float,
        current_end_sec: float,
        lookback_sec: Optional[float] = None,
    ) -> List[Dict]:
        """Run timeline-aware MaxSim over previous+current audio frames.

        The encoded audio spans roughly ``[current_start_sec - lookback_sec,
        current_end_sec]``.  MaxSim windows whose best evidence ends before the
        current vLLM chunk starts are masked out before top-k selection.  This
        preserves boundary-crossing terms while preventing stale terms from the
        previous chunk from leaking into the current prompt.
        """
        if self._audio_buffer is None or len(self._audio_buffer) == 0:
            return []

        k = top_k if top_k is not None else self.top_k
        current_start_sec = max(0.0, float(current_start_sec))
        current_end_sec = max(current_start_sec, float(current_end_sec))
        lookback = self.window_sec if lookback_sec is None else float(lookback_sec)
        lookback = max(0.0, lookback)

        buffer_end = min(len(self._audio_buffer), int(round(current_end_sec * EXPECTED_SAMPLE_RATE)))
        if buffer_end <= 0:
            return []
        encode_start_sec = max(0.0, current_start_sec - lookback)
        buffer_start = min(buffer_end, int(round(encode_start_sec * EXPECTED_SAMPLE_RATE)))
        chunk = self._audio_buffer[buffer_start:buffer_end]
        if len(chunk) == 0:
            return []

        actual_start_sec = float(buffer_start) / EXPECTED_SAMPLE_RATE
        actual_end_sec = float(buffer_end) / EXPECTED_SAMPLE_RATE
        actual_duration = max(1e-6, actual_end_sec - actual_start_sec)

        projected_seq, mask = _encode_audio_projected_seq(
            chunk, self.retriever, self.feat_ext, self.device
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            window_embs = self.retriever._multiscale_pool(projected_seq, mask)
            window_embs = F.normalize(window_embs, p=2, dim=-1).float()

        window_embs_2d = window_embs.squeeze(0)  # [W, D]
        if window_embs_2d.numel() == 0:
            return []
        t_frames = int(projected_seq.shape[1])
        rel_starts, rel_ends = _build_window_time_ranges(
            self.retriever.maxsim_windows, self.retriever.maxsim_stride, t_frames
        )
        if rel_starts.numel() != window_embs_2d.shape[0]:
            logger.warning(
                "Timeline retrieval window count mismatch: ranges=%d embs=%d",
                rel_starts.numel(), window_embs_2d.shape[0],
            )
            return []

        # Map training-time frame seconds to the actual encoded audio span.  This
        # avoids boundary drift from feature-extractor rounding on short chunks.
        nominal_duration = max(float(rel_ends.max().item()), 1e-6)
        scale = actual_duration / nominal_duration
        abs_starts = actual_start_sec + rel_starts.to(self.device) * scale
        abs_ends = actual_start_sec + rel_ends.to(self.device) * scale

        # Keep lookback evidence only when the MaxSim evidence window overlaps
        # the current vLLM chunk timeline.  Windows fully before chunk start are
        # stale; windows crossing the boundary are valid.
        valid_windows = (abs_ends > current_start_sec) & (abs_starts < current_end_sec)
        if int(valid_windows.sum().item()) == 0:
            self._last_results = []
            return []

        sim_by_window = window_embs_2d @ self.text_embs.float().T  # [W, G]
        sim_by_window = sim_by_window.masked_fill(~valid_windows.unsqueeze(1), -float("inf"))
        scores, best_window_idx = sim_by_window.max(dim=0)  # [G], [G]
        finite = torch.isfinite(scores)
        if int(finite.sum().item()) == 0:
            self._last_results = []
            return []

        n = min(int(k), int(finite.sum().item()))
        masked_scores = scores.masked_fill(~finite, -float("inf"))
        top_sco, top_idx = torch.topk(masked_scores, k=n, largest=True, sorted=True)
        top_win = best_window_idx.gather(0, top_idx)
        top_start = abs_starts.gather(0, top_win)
        top_end = abs_ends.gather(0, top_win)

        results = self._build_results(
            top_idx.detach().cpu().numpy(),
            top_sco.detach().cpu().numpy(),
            time_starts=top_start.detach().cpu().numpy(),
            time_ends=top_end.detach().cpu().numpy(),
            retrieval_mode="timeline",
        )
        self._retrieve_offset = buffer_end
        self._last_results = results
        return results

    def retrieve_timeline_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
        lookback_sec: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Batch timeline-aware retrieval for independent streaming states.

        Each request must contain:
          - ``audio_buffer``: accumulated audio up to current_end_sec
          - ``current_start_sec``
          - ``current_end_sec``

        The method batches feature extraction/audio encoding, then applies the
        same per-sample timeline window mask and score-threshold filtering as
        ``retrieve_timeline``.  It intentionally does not mutate the streaming
        retriever's internal audio buffer.
        """
        outputs: List[List[Dict]] = [[] for _ in requests]
        if not requests:
            return outputs

        k = top_k if top_k is not None else self.top_k
        default_lookback = self.window_sec if lookback_sec is None else float(lookback_sec)
        default_lookback = max(0.0, default_lookback)

        chunks: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for req_idx, req in enumerate(requests):
            audio_buffer = np.asarray(req.get("audio_buffer"), dtype=np.float32).flatten()
            if audio_buffer.size == 0:
                continue
            current_start_sec = max(0.0, float(req["current_start_sec"]))
            current_end_sec = max(current_start_sec, float(req["current_end_sec"]))
            cur_lookback = max(0.0, float(req.get("lookback_sec", default_lookback)))

            buffer_end = min(
                len(audio_buffer),
                int(round(current_end_sec * EXPECTED_SAMPLE_RATE)),
            )
            if buffer_end <= 0:
                continue
            encode_start_sec = max(0.0, current_start_sec - cur_lookback)
            buffer_start = min(
                buffer_end,
                int(round(encode_start_sec * EXPECTED_SAMPLE_RATE)),
            )
            chunk = audio_buffer[buffer_start:buffer_end]
            if len(chunk) == 0:
                continue

            actual_start_sec = float(buffer_start) / EXPECTED_SAMPLE_RATE
            actual_end_sec = float(buffer_end) / EXPECTED_SAMPLE_RATE
            chunks.append(chunk)
            metas.append(
                {
                    "request_idx": req_idx,
                    "current_start_sec": current_start_sec,
                    "current_end_sec": current_end_sec,
                    "actual_start_sec": actual_start_sec,
                    "actual_end_sec": actual_end_sec,
                    "actual_duration": max(1e-6, actual_end_sec - actual_start_sec),
                }
            )

        if not chunks:
            return outputs

        batch_t0 = time.perf_counter()
        encode_t0 = time.perf_counter()
        projected_seq, mask = _encode_audio_projected_seq_batch(
            chunks, self.retriever, self.feat_ext, self.device
        )
        encode_s = time.perf_counter() - encode_t0
        post_t0 = time.perf_counter()

        groups_by_frames: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for batch_idx, meta in enumerate(metas):
            valid_frames = int(mask[batch_idx].sum().item())
            if valid_frames <= 0:
                continue
            groups_by_frames.setdefault(valid_frames, []).append((batch_idx, meta))

        text_bank_t = self.text_embs.float().T
        for valid_frames, group_items in groups_by_frames.items():
            group_indices = [batch_idx for batch_idx, _ in group_items]
            seq_g = projected_seq[group_indices, :valid_frames, :]
            mask_g = mask[group_indices, :valid_frames]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                window_embs = self.retriever._multiscale_pool(seq_g, mask_g)
                window_embs = F.normalize(window_embs, p=2, dim=-1).float()

            if window_embs.numel() == 0:
                continue
            rel_starts, rel_ends = _build_window_time_ranges(
                self.retriever.maxsim_windows,
                self.retriever.maxsim_stride,
                valid_frames,
            )
            if rel_starts.numel() != window_embs.shape[1]:
                logger.warning(
                    "Batch timeline retrieval window count mismatch: ranges=%d embs=%d",
                    rel_starts.numel(),
                    window_embs.shape[1],
                )
                continue

            nominal_duration = max(float(rel_ends.max().item()), 1e-6)
            actual_duration = torch.tensor(
                [float(meta["actual_duration"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.device,
            )
            actual_start = torch.tensor(
                [float(meta["actual_start_sec"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.device,
            )
            current_start = torch.tensor(
                [float(meta["current_start_sec"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.device,
            )
            current_end = torch.tensor(
                [float(meta["current_end_sec"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.device,
            )
            scale = actual_duration / nominal_duration
            rel_starts_d = rel_starts.to(self.device)
            rel_ends_d = rel_ends.to(self.device)
            abs_starts = actual_start.unsqueeze(1) + rel_starts_d.unsqueeze(0) * scale.unsqueeze(1)
            abs_ends = actual_start.unsqueeze(1) + rel_ends_d.unsqueeze(0) * scale.unsqueeze(1)
            valid_windows = (
                (abs_ends > current_start.unsqueeze(1))
                & (abs_starts < current_end.unsqueeze(1))
            )
            if int(valid_windows.sum().item()) == 0:
                continue

            sim_by_window = torch.matmul(window_embs, text_bank_t)
            sim_by_window = sim_by_window.masked_fill(
                ~valid_windows.unsqueeze(2),
                -float("inf"),
            )
            scores, best_window_idx = sim_by_window.max(dim=1)
            for row_idx, (_, meta) in enumerate(group_items):
                row_scores = scores[row_idx]
                finite = torch.isfinite(row_scores)
                if int(finite.sum().item()) == 0:
                    continue

                n = min(int(k), int(finite.sum().item()))
                masked_scores = row_scores.masked_fill(~finite, -float("inf"))
                top_sco, top_idx = torch.topk(masked_scores, k=n, largest=True, sorted=True)
                top_win = best_window_idx[row_idx].gather(0, top_idx)
                top_start = abs_starts[row_idx].gather(0, top_win)
                top_end = abs_ends[row_idx].gather(0, top_win)

                outputs[int(meta["request_idx"])] = self._build_results(
                    top_idx.detach().cpu().numpy(),
                    top_sco.detach().cpu().numpy(),
                    time_starts=top_start.detach().cpu().numpy(),
                    time_ends=top_end.detach().cpu().numpy(),
                    retrieval_mode="timeline_batch",
                )

        if os.environ.get("RASST_RAG_PROFILE", "1") == "1":
            print(
                "RASST_RAG_METRIC "
                f"batch_size={len(requests)} encoded={len(chunks)} "
                f"frame_groups={len(groups_by_frames)} terms={len(self.term_list)} "
                f"encode_s={encode_s:.4f} post_s={time.perf_counter() - post_t0:.4f} "
                f"total_s={time.perf_counter() - batch_t0:.4f}",
                flush=True,
            )

        return outputs

    def retrieve_window_with_times(self, top_k: Optional[int] = None) -> List[Dict]:
        """Run the legacy sliding-window retrieve, but attach best-window times.

        This is used by `stride_merge`: intermediate stride windows can still be
        computed ahead of the vLLM call, while final merging can drop candidates
        whose evidence window ended before the current vLLM chunk starts.
        """
        if self._audio_buffer is None or len(self._audio_buffer) == 0:
            return self._last_results

        buffer_end = len(self._audio_buffer)
        new_since_last = buffer_end - self._retrieve_offset
        min_samples = int(0.48 * EXPECTED_SAMPLE_RATE)
        if new_since_last < min_samples:
            return self._last_results

        if self.window_sec > 0:
            window_samples = int(self.window_sec * EXPECTED_SAMPLE_RATE)
            start = max(0, buffer_end - window_samples)
        else:
            start = self._retrieve_offset

        results = self.retrieve_timeline(
            top_k=top_k if top_k is not None else self.top_k,
            current_start_sec=float(start) / EXPECTED_SAMPLE_RATE,
            current_end_sec=float(buffer_end) / EXPECTED_SAMPLE_RATE,
            lookback_sec=0.0,
        )
        for item in results:
            item["retrieval_mode"] = "stride_window"
        self._last_results = results
        return results

    def _build_results(
        self,
        idx_arr: np.ndarray,
        sco_arr: np.ndarray,
        time_starts: Optional[np.ndarray] = None,
        time_ends: Optional[np.ndarray] = None,
        retrieval_mode: str = "window",
    ) -> List[Dict]:
        """Convert top-k index/score arrays into result dicts.

        Applies score_threshold filter: candidates with score < threshold are
        dropped.  Threshold is absolute cosine similarity (MaxSim output).
        """
        results = []
        for pos, (ti, sc) in enumerate(zip(idx_arr, sco_arr)):
            score_f = float(sc)
            if score_f < self.score_threshold:
                continue
            entry = self.term_list[int(ti)]
            translations = entry.get("target_translations", {})
            translation = translations.get(self.target_lang, "")
            if not translation:
                continue
            item = {
                "key": entry.get("key", entry.get("term", "")),
                "term": entry.get("term", ""),
                "translation": translation,
                "score": score_f,
                "retrieval_mode": retrieval_mode,
            }
            if time_starts is not None and time_ends is not None:
                item["time_start"] = float(time_starts[pos])
                item["time_end"] = float(time_ends[pos])
            results.append(item)
        return results

    @staticmethod
    def merge_results(
        result_groups: List[List[Dict]],
        top_k: int,
        min_time_end: Optional[float] = None,
    ) -> List[Dict]:
        """Merge multiple retrieve() results, keeping the highest score per term."""
        best: Dict[str, Dict] = {}
        for results in result_groups:
            for r in results:
                if min_time_end is not None:
                    time_end = r.get("time_end")
                    if time_end is None or float(time_end) <= float(min_time_end):
                        continue
                term = r["term"]
                if term not in best or r["score"] > best[term]["score"]:
                    best[term] = r
        merged = sorted(best.values(), key=lambda x: -x["score"])
        return merged[:top_k]

    def get_current_references(self, min_terms: int = 0) -> List[Dict]:
        """Return the latest retrieval results (for the SimulEval agent)."""
        return self._last_results
