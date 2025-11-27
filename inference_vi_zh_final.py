"""
VI -> ZH Inference Script with Exact Architecture from train_rope_small.py
Filters high-quality translations with BLEU > 60
"""

import os
import math
import random
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
# Configuration - EXACT COPY FROM train_rope_small.py
# =============================================================================

class Config:
    # Data
    train_back_dir = "/home/crl/hienhq/olym/dataset/train_back_maxlen60"
    train_src_file = os.path.join(train_back_dir, "train_interleaved_maxlen60.src")
    train_tgt_file = os.path.join(train_back_dir, "train_interleaved_maxlen60.tgt")
    
    # SentencePiece
    spm_prefix = "./spm_zh_vi_joint"
    vocab_size = 16000
    
    # Model architecture - LARGE WITH RoPE
    d_model = 768          # Increased from 384
    n_heads = 12           # Increased from 6
    num_encoder_layers = 8 # Increased from 4
    num_decoder_layers = 8 # Increased from 4
    d_ff = 3072           # Increased from 1536 (4 * d_model)
    dropout = 0.1         # Keep high dropout for large model
    max_len = 64          # Maximum sequence length
    rope_base = 10000     # RoPE base frequency
    
    # Special tokens
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    
    # Direction control tokens
    zh_to_vi = "<2vi>"
    vi_to_zh = "<2zh>"
    
    # Training config
    batch_size = 32       # Reduced for large model
    learning_rate = 1e-4  # Conservative LR for large model
    warmup_steps = 4000
    max_epochs = 200      # Increased for large model
    accumulation_steps = 2 # Gradient accumulation
    label_smoothing = 0.1
    clip_grad_norm = 1.0
    
    # Eval
    eval_every = 2000
    save_every = 5000
    patience = 10
    
    # Checkpoint
    checkpoint_dir = "checkpoints_continue_zh_vi"
    
    # Vocab special token IDs (0-based) - match train script exactly
    pad_id = 0    # <pad>
    unk_id = 1    # <unk>
    bos_id = 2    # <s>
    eos_id = 3    # </s>
    
    # Random seed
    seed = 42


# =============================================================================
# RMSNorm - EXACT COPY
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
# RoPE (Rotary Position Embedding) - EXACT COPY
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
# FFN with SwiGLU Activation - EXACT COPY
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
# Multi-Head Attention with RoPE - EXACT COPY
# =============================================================================

class MultiHeadAttentionRoPE(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
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
        
        # Linear projections and split heads
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T_q, d_k]
        K = self.W_k(k).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T_k, d_k]
        V = self.W_v(v).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T_k, d_k]
        
        # Apply RoPE to Q and K
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)
        
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
# Encoder Layer (Pre-LN with RoPE) - EXACT COPY
# =============================================================================

class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionRoPE(d_model, n_heads, dropout, rope_base)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            src_pad_mask: [B, T] (True = padding)
        Returns:
            [B, T, d_model]
        """
        # Self-attention with Pre-LN
        h1 = self.ln1(x)
        attn_out = self.self_attn(h1, h1, h1, key_padding_mask=src_pad_mask)
        x = x + self.dropout(attn_out)
        
        # FFN with Pre-LN
        h2 = self.ln2(x)
        ffn_out = self.ffn(h2)
        x = x + self.dropout(ffn_out)
        
        return x


# =============================================================================
# Decoder Layer (Pre-LN with RoPE) - EXACT COPY
# =============================================================================

class DecoderLayer(nn.Module):
    """Pre-LN Transformer decoder layer with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionRoPE(d_model, n_heads, dropout, rope_base)
        
        self.ln2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttentionRoPE(d_model, n_heads, dropout, rope_base)
        
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
            tgt_pad_mask: [B, T_tgt] (True = padding)
            tgt_causal_mask: [T_tgt, T_tgt] (True = mask out)
            src_pad_mask: [B, T_src] (True = padding)
        Returns:
            [B, T_tgt, d_model]
        """
        # Self-attention with Pre-LN
        h1 = self.ln1(y)
        self_attn_out = self.self_attn(
            h1, h1, h1,
            key_padding_mask=tgt_pad_mask,
            attn_mask=tgt_causal_mask
        )
        y = y + self.dropout(self_attn_out)
        
        # Cross-attention with Pre-LN
        h2 = self.ln2(y)
        cross_attn_out = self.cross_attn(
            h2, enc_out, enc_out,
            key_padding_mask=src_pad_mask
        )
        y = y + self.dropout(cross_attn_out)
        
        # FFN with Pre-LN
        h3 = self.ln3(y)
        ffn_out = self.ffn(h3)
        y = y + self.dropout(ffn_out)
        
        return y


# =============================================================================
# Transformer Model - EXACT COPY
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
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout, config.rope_base)
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_final_ln = RMSNorm(config.d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout, config.rope_base)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_final_ln = RMSNorm(config.d_model)
        
        # Output projection (tied with embedding)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def encode(self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        x = self.embedding(src_ids)  # [B, T_src, d_model]
        x = self.emb_dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_pad_mask)
        
        x = self.encoder_final_ln(x)
        return x
    
    def decode(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor,
        src_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Decode target sequence."""
        y = self.embedding(tgt_ids)  # [B, T_tgt, d_model]
        y = self.emb_dropout(y)
        
        for layer in self.decoder_layers:
            y = layer(y, enc_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
        
        y = self.decoder_final_ln(y)
        
        # ===== Output projection (tied weights) =====
        logits = F.linear(y, self.embedding.weight, self.output_bias)
        return logits
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        # Encode
        enc_out = self.encode(src_ids, src_pad_mask)
        
        # Decode
        logits = self.decode(tgt_ids, enc_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
        
        return logits


# =============================================================================
# Inference Functions
# =============================================================================

def create_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create padding mask (True = padding)."""
    return ids == pad_id


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal mask (True = mask out)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def greedy_decode(
    model: TransformerModel,
    src_ids: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    device: torch.device
) -> torch.Tensor:
    """Greedy decoding."""
    model.eval()
    B = src_ids.size(0)
    
    # Create source padding mask
    src_pad_mask = create_padding_mask(src_ids, pad_id)  # [B, T_src]
    
    # Encode source
    with torch.no_grad():
        enc_out = model.encode(src_ids, src_pad_mask)  # [B, T_src, d_model]
    
    # Initialize target with BOS
    tgt_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # [B, 1]
    
    for _ in range(max_len - 1):
        T_tgt = tgt_ids.size(1)
        
        # Create masks
        tgt_pad_mask = create_padding_mask(tgt_ids, pad_id)  # [B, T_tgt]
        tgt_causal_mask = create_causal_mask(T_tgt, device)  # [T_tgt, T_tgt]
        
        # Decode
        with torch.no_grad():
            logits = model.decode(tgt_ids, enc_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
            
        # Get next token
        next_token_logits = logits[:, -1, :]  # [B, vocab_size]
        next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]
        
        # Append to target
        tgt_ids = torch.cat([tgt_ids, next_token_ids], dim=1)  # [B, T_tgt+1]
        
        # Check if all sequences have generated EOS
        if (next_token_ids.squeeze(-1) == eos_id).all():
            break
    
    return tgt_ids


def calculate_bleu(reference: str, candidate: str) -> float:
    """Calculate sentence-level BLEU score."""
    if not sacrebleu:
        return 100.0  # If sacrebleu not available, return high score to include all
    
    if not reference.strip() or not candidate.strip():
        return 0.0
    
    try:
        bleu = sacrebleu.sentence_bleu(candidate, [reference])
        return bleu.score
    except:
        return 0.0


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load SentencePiece model
    spm_model_path = "/home/crl/hienhq/olym/spm_zh_vi_joint.model"
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    print("Loading SentencePiece model...")
    print(f"Vocabulary size: {sp.get_piece_size()}")
    
    # Configuration
    config = Config()
    
    # Load model
    checkpoint_path = "/home/crl/hienhq/olym/checkpoints_continue_zh_vi/checkpoint_epoch_130.pt"
    print(f"Loading model from {checkpoint_path}...")
    
    model = TransformerModel(config, sp.get_piece_size()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset from original data files
    zh_path = "/home/crl/hienhq/nlp/dataset/train/train.zh"
    vi_path = "/home/crl/hienhq/nlp/dataset/train/train.vi"
    print(f"Loading dataset from {zh_path} and {vi_path}...")
    
    zh_sentences = []
    vi_sentences = []
    
    # Load ZH sentences
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh_sentences = [line.strip() for line in f if line.strip()]
    
    # Load VI sentences
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_sentences = [line.strip() for line in f if line.strip()]
    
    # Ensure same length
    min_len = min(len(zh_sentences), len(vi_sentences))
    zh_sentences = zh_sentences[:min_len]
    vi_sentences = vi_sentences[:min_len]
    
    print(f"Loaded {len(zh_sentences)} sentence pairs")
    
    # Create output directory
    output_dir = "/home/crl/hienhq/nlp/dataset/vi_zh_filtered_60"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    output_file = os.path.join(output_dir, "vi_zh_high_quality.txt")
    log_file = os.path.join(output_dir, "translation_log.txt")
    
    high_quality_pairs = []
    total_processed = 0
    bleu_threshold = 60.0
    
    print(f"Starting VI->ZH translation with BLEU threshold {bleu_threshold}...")
    
    with open(log_file, 'w', encoding='utf-8') as log_f:
        for i, (zh_ref, vi_text) in enumerate(tqdm(zip(zh_sentences, vi_sentences), total=len(zh_sentences))):
            try:
                # Prepare input with direction token
                input_text = f"<2zh> {vi_text}"
                input_ids = sp.encode(input_text, out_type=int)
                
                # Add BOS and EOS
                input_ids = [config.bos_id] + input_ids + [config.eos_id]
                
                # Convert to tensor
                src_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                
                # Generate translation
                with torch.no_grad():
                    output_ids = greedy_decode(
                        model, src_tensor, config.max_len, 
                        config.bos_id, config.eos_id, config.pad_id, device
                    )
                
                # Decode output (skip BOS)
                output_tokens = output_ids[0][1:].cpu().tolist()
                if config.eos_id in output_tokens:
                    eos_idx = output_tokens.index(config.eos_id)
                    output_tokens = output_tokens[:eos_idx]
                
                translated_text = sp.decode(output_tokens)
                
                # Calculate BLEU score
                bleu_score = calculate_bleu(zh_ref, translated_text)
                
                # Log the translation
                log_f.write(f"Pair {i+1}:\n")
                log_f.write(f"VI: {vi_text}\n")
                log_f.write(f"ZH_REF: {zh_ref}\n") 
                log_f.write(f"ZH_GEN: {translated_text}\n")
                log_f.write(f"BLEU: {bleu_score:.2f}\n\n")
                
                # Keep high-quality translations
                if bleu_score >= bleu_threshold:
                    high_quality_pairs.append((vi_text, translated_text))
                
                total_processed += 1
                
                # Print progress every 100 sentences
                if (i + 1) % 100 == 0:
                    success_rate = len(high_quality_pairs) / total_processed * 100
                    print(f"Processed: {total_processed}, High-quality: {len(high_quality_pairs)} ({success_rate:.1f}%)")
                
            except Exception as e:
                log_f.write(f"Error processing pair {i+1}: {str(e)}\n\n")
                continue
    
    # Save high-quality pairs
    print(f"\\nSaving {len(high_quality_pairs)} high-quality translations...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for vi, zh in high_quality_pairs:
            f.write(f"{vi}\t{zh}\n")
    
    final_success_rate = len(high_quality_pairs) / total_processed * 100
    print(f"\\nFinal Results:")
    print(f"Total processed: {total_processed}")
    print(f"High-quality pairs (BLEU >= {bleu_threshold}): {len(high_quality_pairs)} ({final_success_rate:.1f}%)")
    print(f"Output saved to: {output_file}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()