#!/usr/bin/env python3
"""PLAN-CL-BI: Bidirectional zh↔vi Transformer with CE + contrastive learning."""

import argparse
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import sacrebleu
except ImportError:  # pragma: no cover - optional dependency
    sacrebleu = None

from inference_continued import InferenceConfig, TransformerModel


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PlanCLBidirectionalConfig(InferenceConfig):
    train_back_dir: str = "/home/crl/hienhq/olym/dataset/train_back_maxlen60"
    train_src_file: str = "/home/crl/hienhq/olym/dataset/train_back_maxlen60/train_interleaved_maxlen60.src"
    train_tgt_file: str = "/home/crl/hienhq/olym/dataset/train_back_maxlen60/train_interleaved_maxlen60.tgt"

    val_ratio: float = 0.03
    max_len: int = 32
    min_len: int = 2

    batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 50

    lr_base: float = 5e-4
    warmup_steps: int = 1500
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    proj_dim: int = 512
    contrastive_tau: float = 0.07
    contrastive_lambda_max: float = 0.015
    contrastive_warmup_steps: int = 1500
    pretrain_epochs: int = 7

    augment_dropout: float = 0.05
    augment_span_mask_prob: float = 0.1
    augment_span_max: int = 2

    save_dir: str = "./checkpoints_plan_cl_bi"
    save_every: int = 5
    eval_max_batches: int = 128

    seed: int = 42


# =============================================================================
# Utility helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_parallel_corpus(src_path: str, tgt_path: str) -> Tuple[List[str], List[str]]:
    if not os.path.isfile(src_path) or not os.path.isfile(tgt_path):
        raise FileNotFoundError("Missing parallel training files.")
    src_lines: List[str] = []
    tgt_lines: List[str] = []
    with open(src_path, "r", encoding="utf-8") as fsrc, open(tgt_path, "r", encoding="utf-8") as ftgt:
        for src_line, tgt_line in zip(fsrc, ftgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line and tgt_line:
                src_lines.append(src_line)
                tgt_lines.append(tgt_line)
    if not src_lines:
        raise ValueError("No training samples found in provided files.")
    return src_lines, tgt_lines


def ensure_tokenizer(config: PlanCLBidirectionalConfig, src_lines: Sequence[str], tgt_lines: Sequence[str]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(config.spm_prefix) or "."
    base_name = os.path.basename(config.spm_prefix) or "spm"
    tokenizer_dir = os.path.join(base_dir, f"tokenizer_plancl_bi_{timestamp}")
    os.makedirs(tokenizer_dir, exist_ok=True)

    new_prefix = os.path.join(tokenizer_dir, base_name)
    print(f"Training SentencePiece tokenizer at {new_prefix}.* ...")
    config.spm_prefix = new_prefix

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for line in src_lines:
            clean = line.replace(config.zh_token, "").replace(config.vi_token, "").strip()
            if clean:
                tmp.write(clean + "\n")
        for line in tgt_lines:
            if line:
                tmp.write(line + "\n")
        corpus_path = tmp.name

    try:
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=config.spm_prefix,
            vocab_size=config.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            user_defined_symbols=f"{config.zh_token},{config.vi_token}",
        )
        print(f"Tokenizer saved to {config.spm_prefix}.[model|vocab]")
    finally:
        os.remove(corpus_path)


def package_tokenizer(prefix: str) -> Dict[str, Any]:
    model_path = f"{prefix}.model"
    vocab_path = f"{prefix}.vocab"
    if not os.path.isfile(model_path) or not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Tokenizer files missing at {prefix}.[model|vocab]")
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    with open(vocab_path, "rb") as f:
        vocab_bytes = f.read()
    return {"prefix": prefix, "model_bytes": model_bytes, "vocab_bytes": vocab_bytes}


def split_pairs(
    src_lines: Sequence[str],
    tgt_lines: Sequence[str],
    val_ratio: float,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    total = min(len(src_lines), len(tgt_lines))
    if total == 0:
        raise ValueError("No paired samples available.")
    split_idx = int((1 - val_ratio) * total)
    if split_idx % 2 != 0:
        split_idx -= 1
    split_idx = max(split_idx, 2)

    train_pairs: List[Tuple[str, str, str]] = []
    for idx in range(split_idx):
        direction = "zh2vi" if idx % 2 == 0 else "vi2zh"
        train_pairs.append((src_lines[idx], tgt_lines[idx], direction))

    val_pairs: List[Tuple[str, str, str]] = []
    for idx in range(split_idx, total):
        if idx % 2 == 0:
            val_pairs.append((src_lines[idx], tgt_lines[idx], "zh2vi"))

    if not val_pairs:
        raise ValueError("Validation split is empty; adjust val_ratio.")
    return train_pairs, val_pairs


# =============================================================================
# Dataset
# =============================================================================


class PlanCLBidirectionalDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str, str]],
        sp_model: spm.SentencePieceProcessor,
        config: PlanCLBidirectionalConfig,
        is_training: bool = True,
    ) -> None:
        self.sp = sp_model
        self.config = config
        self.is_training = is_training

        self.pad_id = self.sp.piece_to_id(config.pad_token)
        self.bos_id = self.sp.piece_to_id(config.bos_token)
        self.eos_id = self.sp.piece_to_id(config.eos_token)

        self.zh_id = self.sp.piece_to_id(config.zh_token)
        self.vi_id = self.sp.piece_to_id(config.vi_token)

        self.samples: List[Dict[str, List[int]]] = []
        self.dropped = 0

        for src_text, tgt_text, direction in pairs:
            lang_tok = config.vi_token if direction == "zh2vi" else config.zh_token
            src_with_token = self.add_lang_token(src_text, lang_tok)
            src_ids = self.sp.encode(src_with_token, out_type=int)
            tgt_ids_plain = self.sp.encode(tgt_text, out_type=int)

            if not self._length_ok(src_ids, tgt_ids_plain):
                self.dropped += 1
                continue

            src_trim = src_ids[: config.max_len]
            tgt_body = tgt_ids_plain[: config.max_len - 2]
            tgt_full = [self.bos_id] + tgt_body + [self.eos_id]

            self.samples.append({
                "src": src_trim,
                "tgt": tgt_full,
            })

        if not self.samples:
            raise ValueError("Dataset empty after filtering; adjust max_len/min_len.")

    def _length_ok(self, src_ids: Sequence[int], tgt_ids: Sequence[int]) -> bool:
        return (
            self.config.min_len <= len(src_ids) <= self.config.max_len
            and self.config.min_len <= len(tgt_ids) <= self.config.max_len
        )

    def add_lang_token(self, text: str, lang_tok: str) -> str:
        text = text.strip()
        if text.startswith(self.config.zh_token) or text.startswith(self.config.vi_token):
            return text
        return f"{lang_tok} {text}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "src": torch.tensor(sample["src"], dtype=torch.long),
            "tgt": torch.tensor(sample["tgt"], dtype=torch.long),
        }


def build_collate_fn(pad_id: int):
    def _pad(seqs: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(seq.size(0) for seq in seqs)
        batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            batch[i, : seq.size(0)] = seq
        return batch

    def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "src": _pad([item["src"] for item in batch]),
            "tgt": _pad([item["tgt"] for item in batch]),
        }

    return collate


# =============================================================================
# Training modules
# =============================================================================


class LabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        mask = (targets != self.ignore_index)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[targets == self.ignore_index] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = loss.masked_fill(~mask, 0.0)
        return loss.sum() / mask.sum().clamp(min=1)


class WarmupInverseSqrtScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, lr_base: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_base = lr_base
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.lr_base * self.step_num / self.warmup_steps
        else:
            lr = self.lr_base * math.sqrt(self.warmup_steps) / math.sqrt(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# =============================================================================
# Augmentation helpers
# =============================================================================


def augment_ids(
    ids: Sequence[int],
    dropout_p: float,
    span_mask_prob: float,
    span_max: int,
    pad_id: int,
    special_ids: Sequence[int],
    unk_id: int,
) -> List[int]:
    special = set(special_ids)
    kept: List[int] = []
    for tok in ids:
        if tok in special:
            kept.append(tok)
            continue
        if random.random() < dropout_p:
            continue
        kept.append(tok)
    if not kept:
        kept = [ids[0]] if ids else [pad_id]
    if span_mask_prob > 0.0 and random.random() < span_mask_prob and len(kept) > 2:
        span_len = random.randint(1, max(1, span_max))
        start = random.randint(0, len(kept) - span_len)
        for i in range(start, min(len(kept), start + span_len)):
            if kept[i] not in special:
                kept[i] = unk_id
    return kept


def augment_batch(
    batch_ids: torch.Tensor,
    config: PlanCLBidirectionalConfig,
    pad_id: int,
    special_ids: Sequence[int],
    unk_id: int,
) -> torch.Tensor:
    augmented: List[List[int]] = []
    for seq in batch_ids.tolist():
        trimmed = [tok for tok in seq if tok != pad_id]
        augmented_seq = augment_ids(
            trimmed,
            dropout_p=config.augment_dropout,
            span_mask_prob=config.augment_span_mask_prob,
            span_max=config.augment_span_max,
            pad_id=pad_id,
            special_ids=special_ids,
            unk_id=unk_id,
        )
        augmented_seq = augmented_seq[: config.max_len]
        if not augmented_seq:
            augmented_seq = [pad_id]
        augmented.append(augmented_seq)

    max_len = max(len(seq) for seq in augmented)
    out = torch.full((len(augmented), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(augmented):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out.to(batch_ids.device)


# =============================================================================
# Contrastive helpers
# =============================================================================


def encode_context(model: TransformerModel, src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    src_pad_mask = (src_ids == pad_id)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)

    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_ln(enc_out)
    return enc_out


def mean_pool(enc_out: torch.Tensor, src_ids: torch.Tensor, pad_id: int, special_ids: Sequence[int]) -> torch.Tensor:
    mask = (src_ids != pad_id)
    for special_id in special_ids:
        mask = mask & (src_ids != special_id)
    mask = mask.float()
    summed = (enc_out * mask.unsqueeze(-1)).sum(dim=1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return summed / denom


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    sim = torch.matmul(z1, z2.transpose(0, 1)) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i = F.cross_entropy(sim, labels)
    loss_j = F.cross_entropy(sim.transpose(0, 1), labels)
    return 0.5 * (loss_i + loss_j)


# =============================================================================
# Decoding helper for BLEU
# =============================================================================


def greedy_decode_vi(
    model: TransformerModel,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: PlanCLBidirectionalConfig,
) -> List[str]:
    model.eval()
    device = src_ids.device
    pad_id = sp_model.piece_to_id(config.pad_token)
    bos_id = sp_model.piece_to_id(config.bos_token)
    eos_id = sp_model.piece_to_id(config.eos_token)

    enc_out = encode_context(model, src_ids, pad_id)
    batch_size = src_ids.size(0)
    tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(config.max_len):
        tgt_pad_mask = (tgt_ids == pad_id)
        tgt_emb = model.embedding(tgt_ids) * model.emb_scale
        tgt_input = model.emb_dropout(tgt_emb)
        tgt_len = tgt_ids.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

        dec_out = tgt_input
        src_pad_mask = (src_ids == pad_id)
        for layer in model.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
        dec_out = model.decoder_final_ln(dec_out)
        logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        if (next_token == eos_id).all():
            break

    outputs = []
    for seq in tgt_ids:
        tokens = seq.tolist()
        if tokens and tokens[0] == bos_id:
            tokens = tokens[1:]
        if eos_id in tokens:
            tokens = tokens[: tokens.index(eos_id)]
        outputs.append(sp_model.decode(tokens))
    return outputs


# =============================================================================
# Training / Evaluation
# =============================================================================


def compute_ce_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    gold = targets[:, 1:]
    logits_flat = logits.reshape(-1, logits.size(-1))
    gold_flat = gold.reshape(-1)
    return criterion(logits_flat, gold_flat)


def train_epoch(
    model: TransformerModel,
    projection: ProjectionHead,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupInverseSqrtScheduler,
    config: PlanCLBidirectionalConfig,
    pad_id: int,
    special_ids: Sequence[int],
    unk_id: int,
    global_step: int,
    pretrain_steps: int,
) -> Tuple[float, float, float, int]:
    model.train()
    projection.train()
    total_ce = 0.0
    total_cl = 0.0
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        src = batch["src"].to(config.device)
        tgt = batch["tgt"].to(config.device)

        optimizer.zero_grad()

        logits = model(src, tgt)
        ce_loss = compute_ce_loss(logits, tgt, criterion)

        if src.size(0) < 2:
            cl_loss = torch.zeros(1, device=config.device)
            lambda_cl = 0.0
        else:
            zh_a = augment_batch(src, config, pad_id, special_ids, unk_id)
            zh_b = augment_batch(src, config, pad_id, special_ids, unk_id)

            enc_a = encode_context(model, zh_a, pad_id)
            enc_b = encode_context(model, zh_b, pad_id)
            z_a = projection(mean_pool(enc_a, zh_a, pad_id, special_ids))
            z_b = projection(mean_pool(enc_b, zh_b, pad_id, special_ids))
            cl_loss = contrastive_loss(z_a, z_b, config.contrastive_tau)

            if global_step < pretrain_steps:
                lambda_cl = 0.0
            else:
                warmed_steps = max(0, global_step - pretrain_steps)
                warmup_ratio = min(1.0, warmed_steps / max(1, config.contrastive_warmup_steps))
                lambda_cl = config.contrastive_lambda_max * warmup_ratio

        loss = ce_loss + lambda_cl * cl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        torch.nn.utils.clip_grad_norm_(projection.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        total_ce += ce_loss.item()
        total_cl += cl_loss.item()
        total_loss += loss.item()

        pbar.set_postfix({
            "ce": f"{ce_loss.item():.3f}",
            "cl": f"{cl_loss.item():.3f}",
            "λ": f"{lambda_cl:.3f}",
            "lr": f"{scheduler.get_lr():.2e}",
        })

    steps = len(dataloader)
    return total_ce / steps, total_cl / steps, total_loss / steps, global_step


def evaluate(
    model: TransformerModel,
    projection: ProjectionHead,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: PlanCLBidirectionalConfig,
    pad_id: int,
    special_ids: Sequence[int],
    unk_id: int,
    global_step: int,
    pretrain_steps: int,
    sp_model: spm.SentencePieceProcessor,
) -> Dict[str, float]:
    model.eval()
    projection.eval()
    total_ce = 0.0
    total_cl = 0.0
    count = 0
    hyps: List[str] = []
    refs: List[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src = batch["src"].to(config.device)
            tgt = batch["tgt"].to(config.device)

            logits = model(src, tgt)
            ce_loss = compute_ce_loss(logits, tgt, criterion)
            total_ce += ce_loss.item()

            if src.size(0) >= 2:
                zh_a = augment_batch(src, config, pad_id, special_ids, unk_id)
                zh_b = augment_batch(src, config, pad_id, special_ids, unk_id)

                enc_a = encode_context(model, zh_a, pad_id)
                enc_b = encode_context(model, zh_b, pad_id)
                z_a = projection(mean_pool(enc_a, zh_a, pad_id, special_ids))
                z_b = projection(mean_pool(enc_b, zh_b, pad_id, special_ids))
                cl_loss = contrastive_loss(z_a, z_b, config.contrastive_tau)
            else:
                cl_loss = torch.zeros(1, device=config.device)

            total_cl += cl_loss.item()
            count += 1

            if sacrebleu is not None and batch_idx < config.eval_max_batches:
                hyps.extend(greedy_decode_vi(model, src, sp_model, config))
                bos_id = sp_model.piece_to_id(config.bos_token)
                eos_id = sp_model.piece_to_id(config.eos_token)
                for tgt_seq in tgt:
                    tokens = tgt_seq.tolist()[1:]
                    if eos_id in tokens:
                        tokens = tokens[: tokens.index(eos_id)]
                    refs.append(sp_model.decode(tokens))

    avg_ce = total_ce / max(1, count)
    avg_cl = total_cl / max(1, count)

    if global_step < pretrain_steps:
        lambda_cl = 0.0
    else:
        warmed_steps = max(0, global_step - pretrain_steps)
        warmup_ratio = min(1.0, warmed_steps / max(1, config.contrastive_warmup_steps))
        lambda_cl = config.contrastive_lambda_max * warmup_ratio

    metrics = {
        "val_ce": avg_ce,
        "val_cl": avg_cl,
        "val_lambda": lambda_cl,
        "val_total": avg_ce + lambda_cl * avg_cl,
    }

    if sacrebleu is not None and hyps and refs:
        metrics["bleu_zh2vi"] = sacrebleu.corpus_bleu(hyps, [refs]).score
    else:
        metrics["bleu_zh2vi"] = 0.0
    return metrics


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan-CL bidirectional zh↔vi training")
    parser.add_argument("--train-src", default=None, help="Path to source training file")
    parser.add_argument("--train-tgt", default=None, help="Path to target training file")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--save-dir", default=None, help="Checkpoint directory override")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--tokenizer-prefix", default=None, help="Base prefix for tokenizer output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PlanCLBidirectionalConfig()
    if args.train_src:
        config.train_src_file = args.train_src
    if args.train_tgt:
        config.train_tgt_file = args.train_tgt
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.save_dir:
        config.save_dir = args.save_dir
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.tokenizer_prefix:
        config.spm_prefix = args.tokenizer_prefix

    os.makedirs(config.save_dir, exist_ok=True)
    set_seed(config.seed)

    print("PLAN-CL bidirectional zh↔vi training")
    print(f"Device: {config.device}")

    src_lines, tgt_lines = load_parallel_corpus(config.train_src_file, config.train_tgt_file)
    total_pairs = min(len(src_lines), len(tgt_lines))
    print(f"Loaded {total_pairs:,} raw pairs")

    ensure_tokenizer(config, src_lines, tgt_lines)

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f"{config.spm_prefix}.model")
    vocab_size = sp_model.GetPieceSize()
    print(f"SentencePiece vocab: {vocab_size}")
    tokenizer_payload = package_tokenizer(config.spm_prefix)

    train_pairs, val_pairs = split_pairs(src_lines, tgt_lines, config.val_ratio)
    print(f"Train pairs (pre-filter): {len(train_pairs):,} | Val pairs (zh→vi): {len(val_pairs):,}")

    train_dataset = PlanCLBidirectionalDataset(train_pairs, sp_model, config, is_training=True)
    val_dataset = PlanCLBidirectionalDataset(val_pairs, sp_model, config, is_training=False)

    print(
        f"Filtered train pairs: {len(train_dataset):,} (dropped {train_dataset.dropped}) | "
        f"Val pairs: {len(val_dataset):,} (dropped {val_dataset.dropped})"
    )

    collate = build_collate_fn(sp_model.piece_to_id(config.pad_token))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )

    model = TransformerModel(config, vocab_size).to(config.device)
    projection = ProjectionHead(config.d_model, config.proj_dim).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    proj_params = sum(p.numel() for p in projection.parameters())
    print(
        f"Model params: {total_params:,} | Projection params: {proj_params:,} | "
        f"Combined: {total_params + proj_params:,}"
    )

    criterion = LabelSmoothedCrossEntropyLoss(
        smoothing=config.label_smoothing,
        ignore_index=sp_model.piece_to_id(config.pad_token),
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=config.lr_base,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupInverseSqrtScheduler(optimizer, config.warmup_steps, config.lr_base)

    pad_id = sp_model.piece_to_id(config.pad_token)
    unk_id = sp_model.piece_to_id(config.unk_token)
    special_ids = [
        pad_id,
        sp_model.piece_to_id(config.bos_token),
        sp_model.piece_to_id(config.eos_token),
        sp_model.piece_to_id(config.zh_token),
        sp_model.piece_to_id(config.vi_token),
    ]

    global_step = 0
    pretrain_steps = config.pretrain_epochs * max(1, len(train_loader))
    best_val_total = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        ce_loss, cl_loss, total_loss, global_step = train_epoch(
            model,
            projection,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            pad_id,
            special_ids,
            unk_id,
            global_step,
            pretrain_steps,
        )
        print(f"Train | CE: {ce_loss:.4f} | CL: {cl_loss:.4f} | Total: {total_loss:.4f}")

        metrics = evaluate(
            model,
            projection,
            val_loader,
            criterion,
            config,
            pad_id,
            special_ids,
            unk_id,
            global_step,
            pretrain_steps,
            sp_model,
        )
        print(
            f"Val  | CE: {metrics['val_ce']:.4f} | CL: {metrics['val_cl']:.4f} | "
            f"λ: {metrics['val_lambda']:.4f} | Total: {metrics['val_total']:.4f} | "
            f"BLEU zh→vi: {metrics['bleu_zh2vi']:.2f}"
        )

        if epoch % config.save_every == 0:
            ckpt_path = os.path.join(config.save_dir, f"plan_cl_bi_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "projection_state_dict": projection.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state": {"step_num": scheduler.step_num},
                    "metrics": metrics,
                    "config": config,
                    "tokenizer": tokenizer_payload,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

        if metrics["val_total"] < best_val_total:
            best_val_total = metrics["val_total"]
            best_path = os.path.join(config.save_dir, "plan_cl_bi_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "projection_state_dict": projection.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state": {"step_num": scheduler.step_num},
                    "metrics": metrics,
                    "config": config,
                    "tokenizer": tokenizer_payload,
                },
                best_path,
            )
            print(f"✓ New best checkpoint (val total={best_val_total:.4f})")


if __name__ == "__main__":
    main()
