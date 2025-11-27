"""
Large Bidirectional Transformer ZH<->VI with RoPE (Rotary Position Embedding)
Complete training script with larger model configuration and RoPE
"""

import os
import math
import random
import argparse
from typing import Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
try:
    import sacrebleu
except ImportError:
    print("Warning: sacrebleu not installed. BLEU score evaluation will be skipped.")
    sacrebleu = None

# =============================================================================
# Configuration - LARGE MODEL WITH RoPE
# =============================================================================

class Config:
    # Data
    train_back_dir = "/home/crl/hienhq/olym/dataset/train_back_maxlen60"
    train_src_file = os.path.join(train_back_dir, "train_interleaved_maxlen60.src")
    train_tgt_file = os.path.join(train_back_dir, "train_interleaved_maxlen60.tgt")
    
    # SentencePiece
    spm_prefix = "./spm_zh_vi_joint"
    vocab_size = 16000
    
    # Model architecture - LARGE WITH RoPE AND GQA
    d_model = 1024          # Increased from 384
    n_heads = 16           # Query heads
    n_kv_heads = 4         # Key-Value heads for GQA (16:4 ratio)
    num_encoder_layers = 12 # Increased from 4
    num_decoder_layers = 12 # Increased from 4
    d_ff = 8192          # Increased from 1536 (4 * d_model)
    dropout = 0.05         # Keep high dropout for large model
    max_len = 64          # Maximum sequence length
    rope_base = 10000     # RoPE base frequency
    
    # Special tokens
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    zh_token = "<2zh>"
    vi_token = "<2vi>"
    
    # Training - Adjusted for larger model
    batch_size = 16       # Reduced due to larger model
    num_epochs = 100
    lr_base = 5e-4        # Reduced for larger model
    warmup_steps = 3000   # Increased warmup
    weight_decay = 0.05   # Keep high weight decay for regularization
    label_smoothing = 0.1
    grad_clip = 1.0
    span_mask_prob = 0.05  # 5% span masking
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8
    
    # Checkpoint
    save_dir = "./checkpoints_rope_771M"
    save_every = 10     # Save every N epochs
    
    # Seed
    seed = 42


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.
        
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return self.weight * x / rms


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================

class RoPE(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, d_model: int, max_len: int = 64, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute frequencies for RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for position encodings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos and sin values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seq_len
            
            # Create position indices
            position = torch.arange(seq_len, device=device, dtype=dtype)  # [T]
            
            # Compute frequencies for each position
            freqs = torch.outer(position, self.inv_freq.to(device))  # [T, d_model//2]
            
            # Compute cos and sin directly
            self._cos_cached = freqs.cos()  # [T, d_model//2]
            self._sin_cached = freqs.sin()  # [T, d_model//2]
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, n_heads, seq_len, d_k]
            seq_len: Sequence length (if None, use x.size(-2))
        Returns:
            cos, sin tensors for RoPE [T, d_k//2] each
        """
        if seq_len is None:
            seq_len = x.size(-2)
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]  # [T, d_k//2] each


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor."""
    # x:   [B, n_heads, seq_len, d_k]
    # cos: [seq_len, d_k//2]
    # sin: [seq_len, d_k//2]
    
    # Split into even and odd dimensions
    x1 = x[..., 0::2]  # [B, n_heads, seq_len, d_k//2]
    x2 = x[..., 1::2]  # [B, n_heads, seq_len, d_k//2]
    
    # Broadcast cos/sin -> [1, 1, seq_len, d_k//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Interleave back to [B, n_heads, seq_len, d_k]
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    
    return rotated_x


# =============================================================================
# FFN with SwiGLU Activation
# =============================================================================

class FFN_SwiGLU(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_ff = d_ff
        
        # Single linear layer outputs 2*d_ff (for gate and value)
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        h = self.linear1(x)  # [B, T, 2*d_ff]
        
        # Split into gate and value
        g, v = h[..., :self.d_ff], h[..., self.d_ff:]
        
        # SwiGLU: g * sigmoid(g) * v
        s = g * torch.sigmoid(g)  # Swish activation
        hidden = s * v  # Gated
        
        out = self.linear2(hidden)  # [B, T, d_model]
        return self.dropout(out)


# =============================================================================
# Multi-Head Attention with RoPE
# =============================================================================

class GroupedQueryAttentionRoPE(nn.Module):
    """Grouped Query Attention with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads  # Query heads
        self.n_kv_heads = n_kv_heads  # Key-Value heads
        self.n_groups = n_heads // n_kv_heads  # How many query heads per KV head
        self.d_k = d_model // n_heads
        self.d_kv = d_model // n_heads  # Same head dimension
        
        # Linear projections
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_kv, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_kv, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # RoPE
        self.rope = RoPE(self.d_k, base=rope_base)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q: [B, T_q, d_model]
            k: [B, T_k, d_model]
            v: [B, T_v, d_model]
            key_padding_mask: [B, T_k] (True = padding)
            attn_mask: [T_q, T_k] (True = mask out)
        Returns:
            [B, T_q, d_model]
        """
        B = q.size(0)
        T_q, T_k = q.size(1), k.size(1)
        
        # Linear projections
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T_q, d_k]
        K = self.W_k(k).view(B, T_k, self.n_kv_heads, self.d_kv).transpose(1, 2)  # [B, n_kv_heads, T_k, d_kv]
        V = self.W_v(v).view(B, T_k, self.n_kv_heads, self.d_kv).transpose(1, 2)  # [B, n_kv_heads, T_k, d_kv]
        
        # Apply RoPE to Q and K
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)
        
        # Expand K and V to match query heads (GQA)
        # Repeat each KV head n_groups times
        K = K.repeat_interleave(self.n_groups, dim=1)  # [B, n_heads, T_k, d_kv]
        V = V.repeat_interleave(self.n_groups, dim=1)  # [B, n_heads, T_k, d_kv]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, T_q, T_k]
        
        # Apply masks
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k] -> [B, 1, 1, T_k]
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        if attn_mask is not None:
            # attn_mask: [T_q, T_k] -> [1, 1, T_q, T_k]
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)  # [B, n_heads, T_q, T_k]
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, n_heads, T_q, d_k]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)  # [B, T_q, d_model]
        
        # Final linear
        return self.W_o(out)


# =============================================================================
# Encoder Layer (Pre-LN with RoPE)
# =============================================================================

class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with RoPE and GQA."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttentionRoPE(d_model, n_heads, n_kv_heads, dropout, rope_base)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T_src, d_model]
            src_pad_mask: [B, T_src]
        Returns:
            [B, T_src, d_model]
        """
        # Self-attention with Pre-LN
        x1 = self.ln1(x)
        attn = self.self_attn(x1, x1, x1, key_padding_mask=src_pad_mask)
        x = x + self.dropout(attn)
        
        # FFN with Pre-LN
        x2 = self.ln2(x)
        ffn_out = self.ffn(x2)
        x = x + ffn_out  # Dropout already in FFN
        
        return x


# =============================================================================
# Decoder Layer (Pre-LN with RoPE)
# =============================================================================

class DecoderLayer(nn.Module):
    """Pre-LN Transformer decoder layer with RoPE and GQA."""
    
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
        """
        Args:
            y: [B, T_tgt, d_model]
            enc_out: [B, T_src, d_model]
            tgt_pad_mask: [B, T_tgt]
            tgt_causal_mask: [T_tgt, T_tgt]
            src_pad_mask: [B, T_src]
        Returns:
            [B, T_tgt, d_model]
        """
        # Masked self-attention with Pre-LN
        y1 = self.ln1(y)
        self_attn = self.self_attn(
            y1, y1, y1,
            key_padding_mask=tgt_pad_mask,
            attn_mask=tgt_causal_mask
        )
        y = y + self.dropout(self_attn)
        
        # Cross-attention with Pre-LN
        y2 = self.ln2(y)
        cross_attn = self.cross_attn(
            y2, enc_out, enc_out,
            key_padding_mask=src_pad_mask
        )
        y = y + self.dropout(cross_attn)
        
        # FFN with Pre-LN
        y3 = self.ln3(y)
        ffn_out = self.ffn(y3)
        y = y + ffn_out  # Dropout already in FFN
        
        return y


# =============================================================================
# Transformer Model (Pre-LN with RoPE)
# =============================================================================

class TransformerModel(nn.Module):
    """Bidirectional Pre-LN Transformer with RoPE for ZH<->VI translation."""
    
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Shared embedding (with padding_idx)
        self.embedding = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx=0  # Assuming pad_id = 0
        )
        
        # Embedding dropout (no positional encoding needed with RoPE)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.n_kv_heads, config.d_ff, config.dropout, config.rope_base)
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_final_ln = RMSNorm(config.d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.n_kv_heads, config.d_ff, config.dropout, config.rope_base)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_final_ln = RMSNorm(config.d_model)
        
        # Output projection (tied with embedding)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # Scale factor for embedding
        self.emb_scale = math.sqrt(config.d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src_ids: [B, T_src] (includes lang token)
            tgt_ids: [B, T_tgt] (includes BOS and EOS)
        Returns:
            logits: [B, T_tgt-1, vocab_size]
        """
        # Create masks
        src_pad_mask = (src_ids == 0)  # [B, T_src]
        tgt_pad_mask = (tgt_ids == 0)  # [B, T_tgt]
        
        # Shift target for decoder input (remove last token)
        tgt_input_ids = tgt_ids[:, :-1]  # [B, T_tgt-1]
        tgt_input_pad_mask = tgt_pad_mask[:, :-1]  # [B, T_tgt-1]
        
        # Causal mask for decoder
        T_tgt_in = tgt_input_ids.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones(T_tgt_in, T_tgt_in, dtype=torch.bool, device=src_ids.device),
            diagonal=1
        )  # [T_tgt-1, T_tgt-1]
        
        # ===== Encoder =====
        # Embedding + scale + dropout (NO positional encoding)
        src_emb = self.embedding(src_ids) * self.emb_scale  # [B, T_src, d_model]
        src_input = self.emb_dropout(src_emb)  # [B, T_src, d_model]
        
        # Encoder layers
        enc_out = src_input
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad_mask)
        
        # Final LN for encoder
        enc_out = self.encoder_final_ln(enc_out)
        
        # ===== Decoder =====
        # Embedding + scale + dropout (NO positional encoding)
        tgt_emb = self.embedding(tgt_input_ids) * self.emb_scale  # [B, T_tgt-1, d_model]
        tgt_input = self.emb_dropout(tgt_emb)  # [B, T_tgt-1, d_model]
        
        # Decoder layers
        dec_out = tgt_input
        for layer in self.decoder_layers:
            dec_out = layer(
                dec_out, enc_out,
                tgt_input_pad_mask, tgt_causal_mask, src_pad_mask
            )
        
        # Final LN for decoder
        dec_out = self.decoder_final_ln(dec_out)
        
        # ===== Output projection (tied weights) =====
        logits = F.linear(dec_out, self.embedding.weight, self.output_bias)
        # [B, T_tgt-1, vocab_size]
        
        return logits


# =============================================================================
# Label Smoothed Cross Entropy Loss
# =============================================================================

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, vocab_size]
            targets: [N]
        Returns:
            scalar loss
        """
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)  # [N, vocab_size]
        
        # Mask for valid positions (not ignore_index)
        mask = (targets != self.ignore_index)
        
        # Create smoothed distribution
        with torch.no_grad():
            # Smoothed probability: (1-smoothing) for true class, smoothing/(V-1) for others
            true_dist = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            
            # Fix: do not generate distribution for PAD tokens to avoid bias
            true_dist[targets == self.ignore_index] = 0.0
        
        # Compute KL divergence: -sum(true_dist * log_probs)
        loss = -(true_dist * log_probs).sum(dim=-1)  # [N]
        
        # Only average over valid positions
        loss = loss.masked_fill(~mask, 0.0)
        return loss.sum() / mask.sum().clamp(min=1)


# =============================================================================
# Dataset
# =============================================================================

class BidirectionalTranslationDataset(Dataset):
    """Bidirectional ZH<->VI dataset."""
    
    def __init__(
        self,
        src_lines: list,
        tgt_lines: list,
        sp_model: spm.SentencePieceProcessor,
        config: Config,
        is_training: bool = True
    ):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.sp = sp_model
        self.config = config
        self.is_training = is_training
        
        # Get special token IDs
        self.pad_id = sp_model.piece_to_id(config.pad_token)
        self.bos_id = sp_model.piece_to_id(config.bos_token)
        self.eos_id = sp_model.piece_to_id(config.eos_token)
        self.unk_id = sp_model.piece_to_id(config.unk_token)
        self.zh_id = sp_model.piece_to_id(config.zh_token)
        self.vi_id = sp_model.piece_to_id(config.vi_token)
        
        # Process samples based on training mode
        self.samples = []
        if is_training:
            # Training: bidirectional (interleaved data)
            for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
                if i % 2 == 0:
                    # Even index: zh->vi (add direction token to source)
                    src_with_token = self.add_lang_token(src, "<2vi>")
                    self.samples.append((src_with_token, tgt, "zh2vi"))
                else:
                    # Odd index: vi->zh (add direction token to source)
                    src_with_token = self.add_lang_token(src, "<2zh>")
                    self.samples.append((src_with_token, tgt, "vi2zh"))
        else:
            # Validation: unidirectional zh->vi only
            for src, tgt in zip(src_lines, tgt_lines):
                # All validation samples are zh->vi
                src_with_token = self.add_lang_token(src, "<2vi>")
                self.samples.append((src_with_token, tgt, "zh2vi"))
    
    def add_lang_token(self, src: str, lang_tok: str) -> str:
        """Add language token if not already present."""
        src = src.strip()
        if src.startswith("<2vi>") or src.startswith("<2zh>"):
            # Already has language token, don't add another
            return src
        return f"{lang_tok} {src}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text, direction = self.samples[idx]
        
        # Encode source and target
        src_ids = self.sp.encode(src_text, out_type=int)
        tgt_ids = [self.bos_id] + self.sp.encode(tgt_text, out_type=int) + [self.eos_id]
        
        # Truncate to MAX_LEN
        src_ids = src_ids[:self.config.max_len]
        tgt_ids = tgt_ids[:self.config.max_len]
        
        # Apply span masking during training
        src_ids = self.apply_span_masking(src_ids, self.is_training)
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
    
    def apply_span_masking(self, src_ids: list, is_training: bool = True) -> list:
        """Apply span masking to source (5% probability)."""
        if not is_training or random.random() > self.config.span_mask_prob:
            return src_ids
        
        src_ids = src_ids.copy()
        
        # Don't mask special tokens
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.zh_id, self.vi_id}
        
        # Find maskable positions
        maskable = [i for i, token_id in enumerate(src_ids) if token_id not in special_ids]
        
        if not maskable:
            return src_ids
        
        # Randomly select span (1-2 tokens)
        num_to_mask = min(random.randint(1, 2), len(maskable))
        if num_to_mask == 0:
            return src_ids
        
        start_idx = random.choice(maskable)
        for i in range(start_idx, min(start_idx + num_to_mask, len(src_ids))):
            if i in maskable:
                # 70% mask with <unk>, 30% keep original to avoid too-hard masking
                if random.random() < 0.7:
                    src_ids[i] = self.unk_id
        
        return src_ids


def collate_fn(batch):
    """Collate function with padding."""
    src_ids_list, tgt_ids_list = zip(*batch)
    
    # Find max lengths
    max_src_len = max(len(s) for s in src_ids_list)
    max_tgt_len = max(len(t) for t in tgt_ids_list)
    
    # Pad sequences
    src_batch = torch.zeros(len(batch), max_src_len, dtype=torch.long)
    tgt_batch = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_ids_list, tgt_ids_list)):
        src_batch[i, :len(src)] = src
        tgt_batch[i, :len(tgt)] = tgt
    
    return src_batch, tgt_batch


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class WarmupInverseSqrtScheduler:
    """Warmup + inverse sqrt learning rate scheduler."""
    
    def __init__(self, optimizer, warmup_steps: int, lr_base: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_base = lr_base
        self.step_num = 0
    
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        
        if self.step_num <= self.warmup_steps:
            lr = self.lr_base * self.step_num / self.warmup_steps
        else:
            lr = self.lr_base * math.sqrt(self.warmup_steps) / math.sqrt(self.step_num)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: TransformerModel,
    dataloader: DataLoader,
    criterion: LabelSmoothedCrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupInverseSqrtScheduler,
    config: Config,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for src_batch, tgt_batch in pbar:
        src_batch = src_batch.to(config.device)
        tgt_batch = tgt_batch.to(config.device)
        
        # Forward pass
        logits = model(src_batch, tgt_batch)  # [B, T_tgt-1, vocab_size]
        
        # Compute loss (target is shifted: tgt[:, 1:])
        targets = tgt_batch[:, 1:]  # [B, T_tgt-1]
        
        logits_flat = logits.reshape(-1, logits.size(-1))  # [B*(T_tgt-1), vocab_size]
        targets_flat = targets.reshape(-1)  # [B*(T_tgt-1)]
        
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_lr():.6f}'
        })
    
    return total_loss / len(dataloader)


@torch.no_grad()
def greedy_decode(
    model: TransformerModel,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: Config,
    max_len: int = 64
) -> list:
    """Greedy decoding for a batch of source sequences."""
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    # Get special token IDs
    bos_id = sp_model.piece_to_id(config.bos_token)
    eos_id = sp_model.piece_to_id(config.eos_token)
    pad_id = sp_model.piece_to_id(config.pad_token)
    
    # Encode source
    src_pad_mask = (src_ids == pad_id)
    
    # Encoder forward pass
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)
    
    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_ln(enc_out)
    
    # Initialize decoder input with BOS tokens
    tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    
    # Generate tokens one by one
    for _ in range(max_len - 1):
        # Current target length
        tgt_len = tgt_ids.size(1)
        
        # Target padding mask and causal mask
        tgt_pad_mask = (tgt_ids == pad_id)
        tgt_causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        
        # Decoder forward pass
        tgt_emb = model.embedding(tgt_ids) * model.emb_scale
        tgt_input = model.emb_dropout(tgt_emb)
        
        dec_out = tgt_input
        for layer in model.decoder_layers:
            dec_out = layer(
                dec_out, enc_out,
                tgt_pad_mask, tgt_causal_mask, src_pad_mask
            )
        dec_out = model.decoder_final_ln(dec_out)
        
        # Output projection
        logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
        
        # Get next tokens (greedy)
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        
        # Append to target sequence
        tgt_ids = torch.cat([tgt_ids, next_tokens], dim=1)
        
        # Check if all sequences have generated EOS
        if (next_tokens.squeeze(-1) == eos_id).all():
            break
    
    # Convert to text (remove BOS/EOS and decode)
    decoded_texts = []
    for seq in tgt_ids:
        # Remove BOS token
        seq = seq[1:].cpu().tolist()
        
        # Remove EOS and everything after
        if eos_id in seq:
            seq = seq[:seq.index(eos_id)]
        
        # Decode
        text = sp_model.decode(seq)
        decoded_texts.append(text)
    
    return decoded_texts


@torch.no_grad()
def evaluate(
    model: TransformerModel,
    dataloader: DataLoader,
    criterion: LabelSmoothedCrossEntropyLoss,
    sp_model: spm.SentencePieceProcessor,
    config: Config
) -> tuple:
    """Evaluate on validation set with loss and BLEU score."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_references = []
    
    for src_batch, tgt_batch in tqdm(dataloader, desc="Evaluating", leave=False):
        src_batch = src_batch.to(config.device)
        tgt_batch = tgt_batch.to(config.device)
        
        # Forward pass for loss
        logits = model(src_batch, tgt_batch)
        
        # Compute loss
        targets = tgt_batch[:, 1:]
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item()
        
        # Generate predictions for BLEU score
        predictions = greedy_decode(model, src_batch, sp_model, config)
        
        # Get reference texts (remove BOS/EOS tokens)
        bos_id = sp_model.piece_to_id(config.bos_token)
        eos_id = sp_model.piece_to_id(config.eos_token)
        
        for tgt_seq in tgt_batch:
            # Remove BOS token and convert to list
            ref_seq = tgt_seq[1:].cpu().tolist()
            
            # Remove EOS and everything after
            if eos_id in ref_seq:
                ref_seq = ref_seq[:ref_seq.index(eos_id)]
            
            # Decode reference
            ref_text = sp_model.decode(ref_seq)
            all_references.append(ref_text)
        
        all_predictions.extend(predictions)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate BLEU score
    bleu_score = 0.0
    if sacrebleu is not None and len(all_predictions) > 0:
        try:
            bleu = sacrebleu.corpus_bleu(all_predictions, [all_references], force=True)
            bleu_score = bleu.score
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")
    
    return avg_loss, bleu_score


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    # Initialize config
    config = Config()
    
    # Set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("LARGE Bidirectional Transformer with RoPE + GQA Training (ZH<->VI)")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}")
    print(f"       GQA ratio: {config.n_heads}:{config.n_kv_heads} ({config.n_heads//config.n_kv_heads}x efficiency)")
    print(f"       num_layers={config.num_encoder_layers}+{config.num_decoder_layers}")
    print(f"       d_ff={config.d_ff}, dropout={config.dropout}")
    print(f"       RoPE base: {config.rope_base}")
    print(f"Training: batch_size={config.batch_size}, epochs={config.num_epochs}")
    print(f"          lr_base={config.lr_base}, warmup={config.warmup_steps}")
    print("=" * 80)
    
    # ========== Step 1: Train SentencePiece (if not exists) ==========
    spm_model_path = f"{config.spm_prefix}.model"
    
    if not os.path.exists(spm_model_path):
        print("\n[1/6] Training SentencePiece model...")
        
        # Combine source and target for joint vocabulary
        temp_corpus = os.path.join(config.save_dir, "temp_corpus.txt")
        
        with open(temp_corpus, "w", encoding="utf-8") as f:
            # Read interleaved data
            with open(config.train_src_file, "r", encoding="utf-8") as fsrc:
                with open(config.train_tgt_file, "r", encoding="utf-8") as ftgt:
                    for src_line, tgt_line in zip(fsrc, ftgt):
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()
                        if src_line and tgt_line:
                            # Remove direction token for vocab training
                            src_clean = src_line.replace("<2vi>", "").replace("<2zh>", "").strip()
                            f.write(src_clean + "\n")
                            f.write(tgt_line + "\n")
        
        # Train SentencePiece
        spm_args = (
            f"--input={temp_corpus} "
            f"--model_prefix={config.spm_prefix} "
            f"--vocab_size={config.vocab_size} "
            f"--model_type=bpe "
            f"--character_coverage=1.0 "
            f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
            f"--user_defined_symbols={config.zh_token},{config.vi_token}"
        )
        spm.SentencePieceTrainer.Train(spm_args)
        
        print(f"✓ SentencePiece model saved to {spm_model_path}")
    else:
        print(f"\n[1/6] Loading existing SentencePiece model from {spm_model_path}")
    
    # Load SentencePiece model
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_model_path)
    vocab_size = sp_model.GetPieceSize()
    print(f"✓ Vocabulary size: {vocab_size}")
    
    # ========== Step 2: Load Data ==========
    print("\n[2/6] Loading training data...")
    
    src_lines = []
    tgt_lines = []
    
    with open(config.train_src_file, "r", encoding="utf-8") as fsrc:
        with open(config.train_tgt_file, "r", encoding="utf-8") as ftgt:
            for src_line, tgt_line in zip(fsrc, ftgt):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if src_line and tgt_line:
                    src_lines.append(src_line)
                    tgt_lines.append(tgt_line)
    
    print(f"✓ Loaded {len(src_lines)} training samples")
    
    # Split train/validation: use most data for training, only 32 zh->vi samples for validation
    # For validation, only keep zh->vi direction (even indices) and limit to 32 samples
    valid_src, valid_tgt = [], []
    valid_count = 0
    for i in range(0, len(src_lines), 2):  # Only even indices (zh->vi)
        if i < len(src_lines) and valid_count < 32:
            valid_src.append(src_lines[i])
            valid_tgt.append(tgt_lines[i])
            valid_count += 1
    
    # Use remaining data for training (skip the 32 validation samples)
    train_src, train_tgt = [], []
    valid_indices = set(range(0, min(64, len(src_lines)), 2))  # First 32 zh->vi pairs (indices 0,2,4...62)
    for i in range(len(src_lines)):
        if i not in valid_indices:
            train_src.append(src_lines[i])
            train_tgt.append(tgt_lines[i])
    
    print(f"  Train: {len(train_src)} samples")
    print(f"  Valid: {len(valid_src)} samples (zh->vi only, limited to 32)")
    
    # ========== Step 3: Create Datasets and Dataloaders ==========
    print("\n[3/6] Creating datasets and dataloaders...")
    
    train_dataset = BidirectionalTranslationDataset(train_src, train_tgt, sp_model, config, is_training=True)
    valid_dataset = BidirectionalTranslationDataset(valid_src, valid_tgt, sp_model, config, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Valid batches: {len(valid_loader)}")
    
    # ========== Step 4: Initialize Model ==========
    print("\n[4/6] Initializing model...")
    
    model = TransformerModel(config, vocab_size).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}")
    
    # ========== Step 5: Setup Training ==========
    print("\n[5/6] Setting up training...")
    
    criterion = LabelSmoothedCrossEntropyLoss(
        smoothing=config.label_smoothing,
        ignore_index=0  # pad_id
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_base,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay
    )
    
    scheduler = WarmupInverseSqrtScheduler(
        optimizer,
        config.warmup_steps,
        config.lr_base
    )
    
    print("✓ Optimizer: AdamW")
    print(f"✓ Scheduler: Warmup ({config.warmup_steps} steps) + Inverse Sqrt")
    print(f"✓ Loss: Label Smoothed CE (smoothing={config.label_smoothing})")
    
    # ========== Step 6: Training Loop ==========
    print("\n[6/6] Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_val_bleu = 0.0
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch)
        
        # Evaluate
        val_loss, val_bleu = evaluate(model, valid_loader, criterion, sp_model, config)
        
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}")
        print(f"  Valid BLEU: {val_bleu:.2f}")
        print(f"  Learning Rate: {scheduler.get_lr():.6f}")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        # Save best model (based on BLEU score - higher is better)
        if val_bleu > best_val_bleu:
            best_val_loss = val_loss
            best_val_bleu = val_bleu
            best_model_path = os.path.join(config.save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'config': config
            }, best_model_path)
            print(f"  ✓ Best model saved! (val_loss: {val_loss:.4f}, val_bleu: {val_bleu:.2f})")
        
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation BLEU: {best_val_bleu:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()