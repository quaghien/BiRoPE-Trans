#!/usr/bin/env python3
"""Bidirectional inference with BLEU logging for the RoPE Transformer."""

import argparse
import os
from typing import List, Tuple

import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm

from inference_continued import (
    Config as TrainingConfig,
    InferenceConfig,
    TransformerModel,
)

# Provide Config symbol in this module so pickled checkpoints referencing
# __main__.Config can be resolved safely when torch.load runs.
Config = TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zh->vi or vi->zh inference with BLEU scoring and CSV export"
    )
    parser.add_argument(
        "--direction",
        choices=["zh2vi", "vi2zh"],
        required=True,
        help="Select translation direction for this run",
    )
    parser.add_argument(
        "--zh-path",
        required=True,
        help="Path to the zh source/reference file (used according to direction)",
    )
    parser.add_argument(
        "--vi-path",
        required=True,
        help="Path to the vi source/reference file (used according to direction)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint (.pt) to load",
    )
    parser.add_argument(
        "--spm-prefix",
        default=None,
        help="SentencePiece prefix (defaults to the value inside InferenceConfig)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV file for detailed results",
    )
    parser.add_argument(
        "--decode-max-len",
        type=int,
        default=None,
        help="Override max decoding length (defaults to config.max_len)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sampled outputs per source sentence (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Sampling temperature (default: 1.5)",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=100,
        help="Top-k filtering for sampling (default: 40)",
    )
    parser.add_argument(
        "--sample-top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) filtering for sampling (default: 0.9)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of sentence pairs processed (for quick tests)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g. cuda, cuda:1, cpu)",
    )
    return parser.parse_args()


def load_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def add_lang_token(text: str, lang_token: str) -> str:
    text = text.strip()
    if text.startswith(lang_token):
        return text
    return f"{lang_token} {text}" if text else lang_token


def compute_bleu(candidate: str, reference: str) -> float:
    if not reference:
        return 0.0
    return sacrebleu.sentence_bleu(candidate, [reference]).score


def encode_source(
    sp_model: spm.SentencePieceProcessor,
    text: str,
    max_len: int,
) -> List[int]:
    token_ids = sp_model.encode(text, out_type=int)
    return token_ids[:max_len]


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Filter logits using top-k and/or top-p (nucleus) sampling."""
    filtered = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, top_k)[0][..., -1, None]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, float("-inf")), filtered)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(filtered, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))

    return filtered


def encode_context(
    model: TransformerModel,
    src_ids: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a single source sequence to reuse across multiple samples."""
    src_pad_mask = (src_ids == pad_id)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)

    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_ln(enc_out)

    return enc_out, src_pad_mask


def sample_translations(
    model: TransformerModel,
    sp_model: spm.SentencePieceProcessor,
    config: InferenceConfig,
    enc_out: torch.Tensor,
    src_pad_mask: torch.Tensor,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_len: int,
) -> List[str]:
    """Draw multiple samples from the decoder using top-k/top-p sampling."""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0 for sampling")

    device = enc_out.device
    bos_id = sp_model.piece_to_id(config.bos_token)
    eos_id = sp_model.piece_to_id(config.eos_token)
    pad_id = sp_model.piece_to_id(config.pad_token)

    samples: List[str] = []
    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [bos_id]

            for _ in range(max_len):
                tgt_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                tgt_pad_mask = (tgt_ids == pad_id)

                T_tgt = tgt_ids.size(1)
                tgt_causal_mask = torch.triu(
                    torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=device),
                    diagonal=1,
                )

                tgt_emb = model.embedding(tgt_ids) * model.emb_scale
                tgt_input = model.emb_dropout(tgt_emb)

                dec_out = tgt_input
                for layer in model.decoder_layers:
                    dec_out = layer(
                        dec_out,
                        enc_out,
                        tgt_pad_mask,
                        tgt_causal_mask,
                        src_pad_mask,
                    )
                dec_out = model.decoder_final_ln(dec_out)

                logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
                next_token_logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k, top_p)

                if torch.isinf(filtered_logits).all():
                    filtered_logits = next_token_logits

                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)

                if next_token == eos_id:
                    break

            content_tokens = tokens[1:]
            if eos_id in content_tokens:
                content_tokens = content_tokens[: content_tokens.index(eos_id)]
            text = sp_model.decode(content_tokens) if content_tokens else ""
            samples.append(text)

    return samples


def prepare_samples(
    zh_lines: List[str],
    vi_lines: List[str],
    direction: str,
) -> Tuple[List[str], List[str]]:
    if len(zh_lines) != len(vi_lines):
        min_len = min(len(zh_lines), len(vi_lines))
        print(
            f"Warning: line count mismatch (zh={len(zh_lines)}, vi={len(vi_lines)}). "
            f"Truncating to {min_len} pairs."
        )
        zh_lines = zh_lines[:min_len]
        vi_lines = vi_lines[:min_len]
    if direction == "zh2vi":
        return zh_lines, vi_lines
    return vi_lines, zh_lines


def main():
    args = parse_args()
    config = InferenceConfig()
    config.device = torch.device(args.device)
    if args.decode_max_len:
        config.max_len = args.decode_max_len
    spm_prefix = args.spm_prefix or config.spm_prefix
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f"{spm_prefix}.model")

    # Instantiate model and load checkpoint
    vocab_size = sp_model.GetPieceSize()
    model = TransformerModel(config, vocab_size).to(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()

    # Prepare data
    zh_lines = load_lines(args.zh_path)
    vi_lines = load_lines(args.vi_path)
    sources, references = prepare_samples(zh_lines, vi_lines, args.direction)
    if args.max_samples is not None:
        sources = sources[: args.max_samples]
        references = references[: args.max_samples]

    lang_token = config.vi_token if args.direction == "zh2vi" else config.zh_token
    input_col = "zh_org" if args.direction == "zh2vi" else "vi_org"
    target_col = "vi_org" if args.direction == "zh2vi" else "zh_org"
    output_prefix = "vi" if args.direction == "zh2vi" else "zh"
    output_cols = [f"{output_prefix}_sample_{i+1}" for i in range(args.num_samples)]
    bleu_cols = [f"bleu_{i+1}" for i in range(args.num_samples)]
    column_names = [input_col, target_col, *output_cols, *bleu_cols, "bleu_avg"]

    pad_id = sp_model.piece_to_id(config.pad_token)

    rows = []
    paired_iter = zip(sources, references)
    for src_text, ref_text in tqdm(paired_iter, total=len(sources), desc="Translating"):
        prepared_src = add_lang_token(src_text, lang_token)
        src_ids = encode_source(sp_model, prepared_src, config.max_len)
        if not src_ids:
            empty_outputs = ["" for _ in range(args.num_samples)]
            zero_bleus = [0.0 for _ in range(args.num_samples)]
            rows.append([src_text, ref_text, *empty_outputs, *zero_bleus, 0.0])
            continue

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=config.device)
        enc_out, src_pad_mask = encode_context(model, src_tensor, pad_id)
        samples = sample_translations(
            model,
            sp_model,
            config,
            enc_out,
            src_pad_mask,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.sample_top_k,
            top_p=args.sample_top_p,
            max_len=config.max_len,
        )
        bleu_scores = [compute_bleu(sample, ref_text) for sample in samples]
        avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
        rows.append([src_text, ref_text, *samples, *bleu_scores, avg_bleu])

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(args.output_csv, index=False)

    avg_bleu = sum(row[-1] for row in rows) / max(len(rows), 1)
    print(f"Saved {len(rows)} rows to {args.output_csv}. Avg BLEU: {avg_bleu:.2f}")


if __name__ == "__main__":
    main()
