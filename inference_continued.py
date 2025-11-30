#!/usr/bin/env python3
"""
Inference script for continued RoPE Transformer model.
Based on train_rope_small.py architecture.
"""

import os
import argparse
import atexit
import math
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Add Config class for checkpoint compatibility
class Config:
    # Data
    train_back_dir = "/home/crl/hienhq/olym/dataset/train_back_maxlen60"
    train_src_file = os.path.join(train_back_dir, "train_interleaved_maxlen60.src")
    train_tgt_file = os.path.join(train_back_dir, "train_interleaved_maxlen60.tgt")
    
    # SentencePiece
    spm_prefix = "/home/crl/hienhq/olym/tokenizer_rope_20251127_053405/spm_zh_vi_joint"
    vocab_size = 8000
    
    # Model architecture - LARGE WITH RoPE
    d_model = 768          # Increased from 384
    n_heads = 12           # Increased from 6
    n_kv_heads = 4         # Grouped KV heads
    num_encoder_layers = 8 # Increased from 4
    num_decoder_layers = 8 # Increased from 4
    d_ff = 3072           # Increased from 1536 (4 * d_model)
    dropout = 0.01        # Match training config
    max_len = 32          # Maximum sequence length
    rope_base = 10000     # RoPE base frequency
    
    # Special tokens
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    zh_token = "<2zh>"
    vi_token = "<2vi>"
    
    # Training hyperparameters
    # batch_size = 16       # Reduced for large model
    # val_batch_size = 8    # Reduced for large model
    # lr = 2e-5            # Reduced learning rate for large model
    warmup_steps = 500
    max_grad_norm = 1.0
    weight_decay = 0.01
    label_smoothing = 0.1
    
    # Training schedule
    epochs = 80
    save_freq = 10        # Save every 10 epochs
    val_freq = 2         # Validate every 2 epochs
    max_checkpoints = 5  # Keep only 5 latest checkpoints
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mixed_precision = True  # Use mixed precision for large model


@dataclass
class InferenceConfig:
    """Configuration for inference - matches train_rope_small.py architecture"""
    # Model architecture - EXACT match with train_rope_small.py
    d_model = 768
    n_heads = 12
    n_kv_heads = 4
    num_encoder_layers = 8
    num_decoder_layers = 8
    d_ff = 3072
    dropout = 0.01
    max_len = 32
    rope_base = 10000.0
    
    # Special tokens
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    zh_token = "<2zh>"
    vi_token = "<2vi>"
    
    # SentencePiece
    spm_prefix = "/home/crl/hienhq/olym/tokenizer_rope_20251127_053405/spm_zh_vi_joint"
    vocab_size = 8000
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Placeholder config classes for checkpoints pickled with custom configs
class PlanCLBidirectionalConfig(InferenceConfig):
    pass


class PlanCLBiCrossConfig(InferenceConfig):
    pass


class PlanCLFineTuneConfig(InferenceConfig):
    """Placeholder for finetune checkpoints pickled with this config class."""
    pass


# =============================================================================
# Checkpoint helpers
# =============================================================================


def apply_checkpoint_config(base_config: InferenceConfig, checkpoint: dict) -> None:
    ckpt_cfg = checkpoint.get("config")
    if ckpt_cfg is None:
        return
    for field, value in vars(base_config).items():
        if hasattr(ckpt_cfg, field):
            setattr(base_config, field, getattr(ckpt_cfg, field))


def materialize_tokenizer(payload: dict) -> Tuple[str, str]:
    if not payload:
        raise ValueError("Tokenizer payload missing or empty in checkpoint.")
    model_bytes = payload.get("model_bytes")
    vocab_bytes = payload.get("vocab_bytes")
    if model_bytes is None or vocab_bytes is None:
        raise ValueError("Tokenizer payload lacks model or vocab bytes.")

    tmp_dir = tempfile.mkdtemp(prefix="spm_from_checkpoint_")
    base_name = os.path.basename(payload.get("prefix", "spm_from_ckpt")) or "spm_from_ckpt"
    prefix = os.path.join(tmp_dir, base_name)

    model_path = f"{prefix}.model"
    vocab_path = f"{prefix}.vocab"
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    with open(vocab_path, "wb") as f:
        f.write(vocab_bytes)
    return prefix, tmp_dir


# =============================================================================
# RMSNorm - Copy from train_rope_small.py
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


# =============================================================================
# RoPE - Copy from train_rope_small.py
# =============================================================================

class RoPE(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, d_model: int, max_len: int = 64, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seq_len
            
            position = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(position, self.inv_freq.to(device))
            
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.size(-2)
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    
    return rotated_x


# =============================================================================
# FFN with SwiGLU - Copy from train_rope_small.py
# =============================================================================

class FFN_SwiGLU(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        g, v = h[..., :self.d_ff], h[..., self.d_ff:]
        s = g * torch.sigmoid(g)
        hidden = s * v
        out = self.linear2(hidden)
        return self.dropout(out)


# =============================================================================
# Grouped Query Attention with RoPE - Copy from train_rope_small.py
# =============================================================================

class GroupedQueryAttentionRoPE(nn.Module):
    """Grouped Query Attention (GQA) with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.d_kv = self.d_k
        
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_kv, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_kv, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.rope = RoPE(self.d_k, base=rope_base)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = q.size(0)
        T_q, T_k = q.size(1), k.size(1)
        
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_kv_heads, self.d_kv).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.n_kv_heads, self.d_kv).transpose(1, 2)
        
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)
        
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_o(out)


# =============================================================================
# Encoder Layer - Copy from train_rope_small.py
# =============================================================================

class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttentionRoPE(d_model, n_heads, n_kv_heads, dropout, rope_base)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.ln1(x)
        attn = self.self_attn(x1, x1, x1, key_padding_mask=src_pad_mask)
        x = x + self.dropout(attn)
        
        x2 = self.ln2(x)
        ffn_out = self.ffn(x2)
        x = x + ffn_out
        
        return x


# =============================================================================
# Decoder Layer - Copy from train_rope_small.py
# =============================================================================

class DecoderLayer(nn.Module):
    """Pre-LN Transformer decoder layer with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttentionRoPE(d_model, n_heads, n_kv_heads, dropout, rope_base)
        
        self.ln2 = RMSNorm(d_model)
        self.cross_attn = GroupedQueryAttentionRoPE(d_model, n_heads, n_kv_heads, dropout, rope_base)
        
        self.ln3 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        y: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_pad_mask: Optional[torch.Tensor] = None,
        tgt_causal_mask: Optional[torch.Tensor] = None,
        src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y1 = self.ln1(y)
        self_attn = self.self_attn(
            y1, y1, y1,
            key_padding_mask=tgt_pad_mask,
            attn_mask=tgt_causal_mask
        )
        y = y + self.dropout(self_attn)
        
        y2 = self.ln2(y)
        cross_attn = self.cross_attn(
            y2, enc_out, enc_out,
            key_padding_mask=src_pad_mask
        )
        y = y + self.dropout(cross_attn)
        
        y3 = self.ln3(y)
        ffn_out = self.ffn(y3)
        y = y + ffn_out
        
        return y


# =============================================================================
# Transformer Model - Copy from train_rope_small.py
# =============================================================================

class TransformerModel(nn.Module):
    """Bidirectional Pre-LN Transformer with RoPE for ZH<->VI translation."""
    
    def __init__(self, config: InferenceConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.d_ff,
                config.dropout,
                config.rope_base
            )
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_final_ln = RMSNorm(config.d_model)
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.d_ff,
                config.dropout,
                config.rope_base
            )
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_final_ln = RMSNorm(config.d_model)
        
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        
        self.emb_scale = math.sqrt(config.d_model)
    
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_pad_mask = (src_ids == 0)
        tgt_pad_mask = (tgt_ids == 0)
        
        tgt_input_ids = tgt_ids[:, :-1]
        tgt_input_pad_mask = tgt_pad_mask[:, :-1]
        
        T_tgt_in = tgt_input_ids.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones(T_tgt_in, T_tgt_in, dtype=torch.bool, device=src_ids.device),
            diagonal=1
        )
        
        # Encoder
        src_emb = self.embedding(src_ids) * self.emb_scale
        src_input = self.emb_dropout(src_emb)
        
        enc_out = src_input
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad_mask)
        
        enc_out = self.encoder_final_ln(enc_out)
        
        # Decoder
        tgt_emb = self.embedding(tgt_input_ids) * self.emb_scale
        tgt_input = self.emb_dropout(tgt_emb)
        
        dec_out = tgt_input
        for layer in self.decoder_layers:
            dec_out = layer(
                dec_out, enc_out,
                tgt_input_pad_mask, tgt_causal_mask, src_pad_mask
            )
        
        dec_out = self.decoder_final_ln(dec_out)
        
        # Output projection (tied weights)
        logits = F.linear(dec_out, self.embedding.weight, self.output_bias)
        
        return logits


# =============================================================================
# Beam Search Decoding
# =============================================================================

@dataclass
class BeamSearchHypothesis:
    """A single hypothesis in beam search."""
    tokens: List[int]
    log_prob: float
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob


def beam_search_decode(
    model: TransformerModel,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: InferenceConfig,
    beam_size: int = 3,
    max_len: int = 64,
    length_penalty: float = 0.6,
    top_k: int = 5
) -> List[str]:
    """Beam search decoding for a batch of source sequences."""
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    # Get special token IDs
    bos_id = sp_model.piece_to_id(config.bos_token)
    eos_id = sp_model.piece_to_id(config.eos_token)
    pad_id = sp_model.piece_to_id(config.pad_token)
    
    results = []
    
    with torch.no_grad():
        # Encode source
        src_pad_mask = (src_ids == pad_id)
        src_emb = model.embedding(src_ids) * model.emb_scale
        src_input = model.emb_dropout(src_emb)
        
        enc_out = src_input
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_pad_mask)
        enc_out = model.encoder_final_ln(enc_out)
        
        # Beam search for each sample in batch
        for b in range(batch_size):
            # Initialize beam with BOS token
            beams = [BeamSearchHypothesis([bos_id], 0.0)]
            finished_beams = []
            
            # Current encoder output for this sample
            curr_enc_out = enc_out[b:b+1]  # [1, T_src, d_model]
            curr_src_mask = src_pad_mask[b:b+1]  # [1, T_src]
            
            for step in range(max_len):
                if not beams:
                    break
                
                new_beams = []
                
                for beam in beams:
                    if beam.tokens[-1] == eos_id:
                        finished_beams.append(beam)
                        continue
                    
                    # Prepare decoder input
                    tgt_ids = torch.tensor(beam.tokens, device=device).unsqueeze(0)  # [1, T]
                    tgt_pad_mask = (tgt_ids == pad_id)
                    
                    T_tgt = tgt_ids.size(1)
                    tgt_causal_mask = torch.triu(
                        torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=device),
                        diagonal=1
                    )
                    
                    # Decoder forward pass
                    tgt_emb = model.embedding(tgt_ids) * model.emb_scale
                    tgt_input = model.emb_dropout(tgt_emb)
                    
                    dec_out = tgt_input
                    for layer in model.decoder_layers:
                        dec_out = layer(
                            dec_out, curr_enc_out,
                            tgt_pad_mask, tgt_causal_mask, curr_src_mask
                        )
                    dec_out = model.decoder_final_ln(dec_out)
                    
                    # Output projection
                    logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
                    
                    # Apply top-k filtering if specified
                    if top_k is not None and top_k > 0:
                        top_k_probs, top_k_indices = torch.topk(log_probs, min(top_k, log_probs.size(-1)))
                        # Create a mask for top-k tokens
                        mask = torch.full_like(log_probs, float('-inf'))
                        mask.scatter_(1, top_k_indices, top_k_probs)
                        log_probs = mask
                    
                    # Get top beam_size candidates
                    top_probs, top_indices = torch.topk(log_probs.squeeze(0), beam_size)
                    
                    for i in range(beam_size):
                        next_token = top_indices[i].item()
                        next_log_prob = top_probs[i].item()
                        
                        new_tokens = beam.tokens + [next_token]
                        new_log_prob = beam.log_prob + next_log_prob
                        
                        new_beams.append(BeamSearchHypothesis(new_tokens, new_log_prob))
                
                # Keep only top beam_size beams
                beams = sorted(new_beams, key=lambda x: x.log_prob, reverse=True)[:beam_size]
            
            # Add remaining beams to finished
            finished_beams.extend(beams)
            
            # Select best beam with length penalty
            if finished_beams:
                best_beam = max(finished_beams, key=lambda x: x.log_prob / (len(x.tokens) ** 0.6))
            else:
                # Fallback if no beams finished
                best_beam = beams[0] if beams else BeamSearchHypothesis([bos_id, eos_id], 0.0)
            
            # Decode to text
            tokens = best_beam.tokens[1:]  # Remove BOS
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]  # Remove EOS and after
            
            text = sp_model.decode(tokens)
            results.append(text)
    
    return results


# =============================================================================
# Main Inference Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Inference with continued RoPE Transformer')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/crl/hienhq/olym/contrastive_v1/v1_best.pt',
                       help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, choices=['public', 'private'], default='public',
                       help='Which test dataset to use')

    parser.add_argument('--output_csv', type=str, default=None,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Fixed inference parameters
    beam_size = 3
    top_k = 5
    length_penalty = 0.6
    
    # Initialize config and load checkpoint
    config = InferenceConfig()
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    apply_checkpoint_config(config, checkpoint)

    # Load SentencePiece model (prefer packaged tokenizer)
    sp_model = spm.SentencePieceProcessor()
    temp_tokenizer_dir = None
    if "tokenizer" in checkpoint:
        try:
            config.spm_prefix, temp_tokenizer_dir = materialize_tokenizer(checkpoint["tokenizer"])
            atexit.register(lambda: shutil.rmtree(temp_tokenizer_dir, ignore_errors=True))
        except ValueError as exc:
            print(f"[warning] Failed to materialize tokenizer from checkpoint: {exc}. Falling back to {config.spm_prefix}.")

    spm_model_path = f"{config.spm_prefix}.model"
    if not os.path.isfile(spm_model_path):
        raise FileNotFoundError(f"SentencePiece model not found at {spm_model_path}. Provide --tokenizer-prefix or ensure checkpoint packaged tokenizer.")
    sp_model.Load(spm_model_path)
    vocab_size = sp_model.GetPieceSize()

    # Create model
    model = TransformerModel(config, vocab_size).to(config.device)
    
    # Load checkpoint weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Load test data
    if args.dataset == 'public':
        test_file = "/home/crl/hienhq/olym/dataset/public_test/public_test.zh"
    else:
        test_file = "/home/crl/hienhq/olym/dataset/private_test/private_test.zh"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    # Process one sample at a time
    all_results = []
    model.eval()
    
    for sentence in tqdm(test_sentences, desc="Translating"):
        # Add <2vi> direction token
        src_with_token = f"<2vi> {sentence}"
        src_ids = sp_model.encode(src_with_token, out_type=int)
        src_ids = src_ids[:config.max_len]  # Truncate if needed
        
        # Prepare single sample batch
        src_batch = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(config.device)
        
        # Generate translation
        translations = beam_search_decode(
            model, src_batch, sp_model, config,
            beam_size=beam_size,
            max_len=config.max_len,
            length_penalty=length_penalty,
            top_k=top_k
        )
        
        # Store result
        all_results.append({
            'tieng_trung': sentence,
            'tieng_viet': translations[0]
        })
    
    # Save results
    if args.output_csv is None:
        args.output_csv = f"{args.dataset}_{args.checkpoint.split('/')[-1].replace('.pt', '')}.csv"
    
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False, encoding='utf-8')


if __name__ == "__main__":
    main()