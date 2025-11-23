#!/usr/bin/env python3
"""
Inference script for RoPE Transformer model with beam search.
Generates predictions for different top_k values.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import pandas as pd
from typing import List, Tuple, Optional
import math
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for RoPE Transformer."""
    # Model architecture
    d_model = 768
    n_heads = 12
    d_ff = 3072  # 4 * d_model
    num_encoder_layers = 8
    num_decoder_layers = 8
    dropout = 0.3
    max_len = 64
    rope_base = 10000.0
    
    # Training
    batch_size = 32
    lr = 1e-4
    warmup_steps = 4000
    label_smoothing = 0.1
    weight_decay = 0.05
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin values
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
        
        return self.linear2(self.dropout(hidden))


# =============================================================================
# Multi-Head Attention with RoPE
# =============================================================================

class MultiHeadAttentionRoPE(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear transformations
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE for positional encoding
        self.rope = RoPE(self.d_k, base=rope_base)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = key.size(1)
        
        # Linear transformations
        Q = self.W_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T_q, d_k]
        K = self.W_k(key).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)    # [B, n_heads, T_k, d_k]
        V = self.W_v(value).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T_k, d_k]
        
        # Apply RoPE to Q and K
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, T_q, T_k]
        
        # Apply attention mask
        if attn_mask is not None:
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
    """Pre-LN Transformer encoder layer with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionRoPE(d_model, n_heads, dropout, rope_base)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN: Self-attention
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x, src_pad_mask)
        x = residual + self.dropout(x)
        
        # Pre-LN: FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


# =============================================================================
# Decoder Layer (Pre-LN with RoPE)
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
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN: Self-attention
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = residual + self.dropout(x)
        
        # Pre-LN: Cross-attention
        residual = x
        x = self.ln2(x)
        x = self.cross_attn(x, enc_output, enc_output, src_pad_mask)
        x = residual + self.dropout(x)
        
        # Pre-LN: FFN
        residual = x
        x = self.ln3(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


# =============================================================================
# Transformer Model
# =============================================================================

class Transformer(nn.Module):
    """Transformer model with RoPE."""
    
    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Embeddings (shared between input and output)
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout, config.rope_base)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Final layer norm for encoder
        self.encoder_final_ln = RMSNorm(config.d_model)
        
        # Decoder layers
        decoder_layer = DecoderLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout, config.rope_base)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_final_ln = RMSNorm(config.d_model)
        
        # Output projection (tied with embedding)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # Scale factor for embedding
        self.d_model = config.d_model
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embedding with scaling
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Encoder
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad_mask)
        
        enc_out = self.encoder_final_ln(enc_out)
        
        # Decoder
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_pad_mask)
        
        dec_out = self.decoder_final_ln(dec_out)
        
        # Output projection (tied weights)
        logits = F.linear(dec_out, self.embedding.weight, self.output_bias)
        
        return logits


# =============================================================================
# Beam Search Implementation
# =============================================================================

class BeamSearchHypothesis:
    """Single hypothesis in beam search."""
    def __init__(self, tokens: List[int], log_prob: float):
        self.tokens = tokens
        self.log_prob = log_prob
        self.finished = False
    
    def extend(self, token: int, log_prob: float):
        """Extend hypothesis with new token."""
        new_tokens = self.tokens + [token]
        new_log_prob = self.log_prob + log_prob
        return BeamSearchHypothesis(new_tokens, new_log_prob)
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_pad_mask: Optional[torch.Tensor],
    sp_model: spm.SentencePieceProcessor,
    beam_size: int = 3,
    max_length: int = 64,
    top_k: Optional[int] = None
) -> str:
    """Beam search decoding with top-k sampling."""
    device = src.device
    bos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()
    pad_id = sp_model.pad_id()
    
    # Encode source
    model.eval()
    with torch.no_grad():
        # Encoder
        src_emb = model.embedding(src) * math.sqrt(model.d_model)
        enc_out = src_emb
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_pad_mask)
        enc_out = model.encoder_final_ln(enc_out)
    
    # Initialize beam with BOS token
    initial_hyp = BeamSearchHypothesis([bos_id], 0.0)
    beam = [initial_hyp]
    finished_hyps = []
    
    for step in range(max_length):
        if not beam:
            break
            
        # Current hypotheses
        current_hyps = []
        
        for hyp in beam:
            if hyp.finished:
                finished_hyps.append(hyp)
                continue
                
            # Prepare decoder input
            tgt_tokens = torch.tensor([hyp.tokens], device=device)
            tgt_len = len(hyp.tokens)
            
            # Causal mask
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
            
            with torch.no_grad():
                # Decoder forward
                tgt_emb = model.embedding(tgt_tokens) * math.sqrt(model.d_model)
                dec_out = tgt_emb
                for layer in model.decoder_layers:
                    dec_out = layer(dec_out, enc_out, tgt_mask, src_pad_mask)
                dec_out = model.decoder_final_ln(dec_out)
                
                # Output projection
                logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)  # [vocab_size]
                
                # Top-k sampling if specified
                if top_k is not None and top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(log_probs, min(top_k, log_probs.size(-1)))
                    # Extend with top-k candidates
                    for prob, idx in zip(top_k_probs, top_k_indices):
                        token = idx.item()
                        new_hyp = hyp.extend(token, prob.item())
                        if token == eos_id:
                            new_hyp.finished = True
                        current_hyps.append(new_hyp)
                else:
                    # Use all vocabulary
                    for token in range(log_probs.size(-1)):
                        prob = log_probs[token].item()
                        new_hyp = hyp.extend(token, prob)
                        if token == eos_id:
                            new_hyp.finished = True
                        current_hyps.append(new_hyp)
        
        # Keep best beam_size hypotheses
        current_hyps.sort(reverse=True)  # Sort by log_prob (descending)
        beam = current_hyps[:beam_size]
    
    # Add remaining hypotheses to finished
    finished_hyps.extend(beam)
    
    # Return best hypothesis
    if finished_hyps:
        best_hyp = max(finished_hyps)
        # Remove BOS token and EOS if present
        tokens = best_hyp.tokens[1:]  # Remove BOS
        if tokens and tokens[-1] == eos_id:
            tokens = tokens[:-1]  # Remove EOS
        return sp_model.decode(tokens)
    
    return ""


# =============================================================================
# Inference Functions
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Load SentencePiece model
    sp_model = spm.SentencePieceProcessor()
    sp_model.load('spm_zh_vi_joint.model')
    vocab_size = sp_model.get_piece_size()
    
    # Create model
    model = Transformer(vocab_size, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Validation BLEU: {checkpoint.get('val_bleu', 'N/A')}")
    
    return model, sp_model, config


def process_test_data(file_path: str, sp_model: spm.SentencePieceProcessor, max_len: int = 64):
    """Process test data file."""
    print(f"Processing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    processed_data = []
    for line in lines:
        if line:  # Skip empty lines
            # Add language token and tokenize
            src_with_token = f"<2vi> {line}"
            src_tokens = sp_model.encode(src_with_token, add_bos=True, add_eos=True)
            
            # Truncate if too long
            if len(src_tokens) > max_len:
                src_tokens = src_tokens[:max_len-1] + [sp_model.eos_id()]
            
            processed_data.append({
                'original': line,
                'tokens': src_tokens
            })
    
    print(f"Processed {len(processed_data)} samples")
    return processed_data


def run_inference(
    model: nn.Module,
    sp_model: spm.SentencePieceProcessor,
    test_data: List[dict],
    top_k: Optional[int] = None,
    beam_size: int = 3,
    device: torch.device = None
):
    """Run inference on test data."""
    predictions = []
    
    print(f"Running inference with top_k={top_k}, beam_size={beam_size}")
    
    for i, sample in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples")
        
        # Prepare input
        src_tokens = torch.tensor([sample['tokens']], device=device)
        src_pad_mask = (src_tokens == sp_model.pad_id())
        
        # Generate prediction
        prediction = beam_search_decode(
            model=model,
            src=src_tokens,
            src_pad_mask=src_pad_mask,
            sp_model=sp_model,
            beam_size=beam_size,
            top_k=top_k
        )
        
        predictions.append({
            'tieng_trung': sample['original'],
            'tieng_viet': prediction
        })
    
    return predictions


def save_predictions(predictions: List[dict], output_path: str):
    """Save predictions to CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved predictions to {output_path}")


def main():
    """Main inference function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RoPE Transformer Inference with configurable top_k')
    parser.add_argument('--top_k', type=str, default=None, 
                       help='Top-k value for sampling (use "None" for no filtering)')
    parser.add_argument('--beam_size', type=int, default=3, 
                       help='Beam size for beam search (default: 3)')
    parser.add_argument('--checkpoint', type=str, 
                       default="/home/crl/hienhq/olym/checkpoints_rope/checkpoint_epoch_70.pt",
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint
    
    # Parse top_k (handle None string)
    if args.top_k is None or args.top_k.lower() == 'none':
        top_k = None
    else:
        top_k = int(args.top_k)
    
    beam_size = args.beam_size
    
    # Determine output suffix
    if top_k is None:
        top_k_name = "full"
    else:
        top_k_name = f"top{top_k}"
    
    # Input paths
    public_test_path = "/home/crl/hienhq/olym/dataset/public_test/public_test.zh"
    private_test_path = "/home/crl/hienhq/olym/dataset/private_test/private_test.zh"
    
    print("=" * 80)
    print("RoPE Transformer Inference")
    print(f"Top-k: {top_k}")
    print(f"Beam size: {beam_size}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Load model
    model, sp_model, config = load_model(checkpoint_path, device)
    
    # Process test data
    print("\nProcessing test data...")
    public_data = process_test_data(public_test_path, sp_model, config.max_len)
    private_data = process_test_data(private_test_path, sp_model, config.max_len)
    
    # Run inference
    print(f"\n" + "=" * 60)
    print(f"Running inference with top_k={top_k} ({top_k_name})")
    print("=" * 60)
    
    # Public test
    print("\nPublic test:")
    public_predictions = run_inference(
        model, sp_model, public_data, top_k, beam_size, device
    )
    public_output = f"public_test_{top_k_name}.csv"
    save_predictions(public_predictions, public_output)
    
    # Private test
    print("\nPrivate test:")
    private_predictions = run_inference(
        model, sp_model, private_data, top_k, beam_size, device
    )
    private_output = f"private_test_{top_k_name}.csv"
    save_predictions(private_predictions, private_output)
    
    print(f"\n" + "=" * 80)
    print(f"Inference completed for {top_k_name}!")
    print("Generated files:")
    print(f"  - {public_output}")
    print(f"  - {private_output}")
    print("=" * 80)


if __name__ == "__main__":
    main()