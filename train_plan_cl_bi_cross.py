#!/usr/bin/env python3
"""PLAN-CL-BI-CROSS: Bidirectional zh↔vi CE + cross-lingual contrastive training."""

import argparse
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


@dataclass
class PlanCLBiCrossConfig(InferenceConfig):
    train_src_file: str = "/home/crl/hienhq/olym/dataset/train_back_maxlen60/train_interleaved_maxlen60.src"
    train_tgt_file: str = "/home/crl/hienhq/olym/dataset/train_back_maxlen60/train_interleaved_maxlen60.tgt"

    val_ratio: float = 0.03
    max_len: int = 32
    min_len: int = 2

    batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 40

    lr_base: float = 5e-4
    warmup_steps: int = 1500
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    proj_dim: int = 512
    contrastive_tau: float = 0.07
    pretrain_epochs: int = 8
    cross_lambda_max: float = 0.008
    cross_warmup_steps: int = 2000

    save_dir: str = "./checkpoints_plan_cl_bi_cross"
    save_every: int = 5
    eval_max_batches: int = 128

    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_interleaved_samples(src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
    if not os.path.isfile(src_path) or not os.path.isfile(tgt_path):
        raise FileNotFoundError("Missing parallel training files.")
    samples: List[Tuple[str, str]] = []
    with open(src_path, "r", encoding="utf-8") as fsrc, open(tgt_path, "r", encoding="utf-8") as ftgt:
        for idx, (src_line, tgt_line) in enumerate(zip(fsrc, ftgt)):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if not src_line or not tgt_line:
                continue
            if idx % 2 == 0:
                zh, vi = src_line, tgt_line
            else:
                zh, vi = tgt_line, src_line
            samples.append((zh.strip(), vi.strip()))
    if not samples:
        raise ValueError("No bilingual samples produced from interleaved data.")
    return samples


def ensure_tokenizer(config: PlanCLBiCrossConfig, samples: Sequence[Tuple[str, str]]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(config.spm_prefix) or "."
    base_name = os.path.basename(config.spm_prefix) or "spm"
    tokenizer_dir = os.path.join(base_dir, f"tokenizer_plancl_bi_cross_{timestamp}")
    os.makedirs(tokenizer_dir, exist_ok=True)

    new_prefix = os.path.join(tokenizer_dir, base_name)
    print(f"Training SentencePiece tokenizer at {new_prefix}.* ...")
    config.spm_prefix = new_prefix

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for zh, vi in samples:
            tmp.write(zh.replace(config.zh_token, "").replace(config.vi_token, "").strip() + "\n")
            tmp.write(vi.replace(config.zh_token, "").replace(config.vi_token, "").strip() + "\n")
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
    if not (os.path.isfile(model_path) and os.path.isfile(vocab_path)):
        raise FileNotFoundError(f"Tokenizer files missing at {prefix}.[model|vocab]")
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    with open(vocab_path, "rb") as f:
        vocab_bytes = f.read()
    return {"prefix": prefix, "model_bytes": model_bytes, "vocab_bytes": vocab_bytes}


def split_samples(
    samples: Sequence[Tuple[str, str]],
    val_ratio: float,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    total = len(samples)
    val_size = max(1, int(total * val_ratio))
    train_samples = list(samples[val_size:])
    val_samples = list(samples[:val_size])
    if not train_samples:
        raise ValueError("Training split empty; reduce val_ratio.")
    return train_samples, val_samples


class PlanCLBiCrossDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, str]],
        sp_model: spm.SentencePieceProcessor,
        config: PlanCLBiCrossConfig,
        include_dirs: Sequence[str],
    ) -> None:
        self.sp = sp_model
        self.config = config
        self.include_dirs = set(include_dirs)

        self.pad_id = self.sp.piece_to_id(config.pad_token)
        self.bos_id = self.sp.piece_to_id(config.bos_token)
        self.eos_id = self.sp.piece_to_id(config.eos_token)

        self.instances: List[Dict[str, Any]] = []
        self.dropped = 0

        for zh_text, vi_text in samples:
            zh_tokens = self.sp.encode(zh_text, out_type=int)
            vi_tokens = self.sp.encode(vi_text, out_type=int)
            if not self._length_ok(zh_tokens) or not self._length_ok(vi_tokens):
                self.dropped += 1
                continue

            if "zh2vi" in self.include_dirs:
                src = self.sp.encode(f"{config.vi_token} {zh_text}", out_type=int)[: config.max_len]
                tgt = [self.bos_id] + vi_tokens[: config.max_len - 2] + [self.eos_id]
                self.instances.append({
                    "src": src,
                    "tgt": tgt,
                    "direction": "zh2vi",
                    "zh_text": zh_text,
                    "vi_text": vi_text,
                })

            if "vi2zh" in self.include_dirs:
                src = self.sp.encode(f"{config.zh_token} {vi_text}", out_type=int)[: config.max_len]
                tgt = [self.bos_id] + zh_tokens[: config.max_len - 2] + [self.eos_id]
                self.instances.append({
                    "src": src,
                    "tgt": tgt,
                    "direction": "vi2zh",
                    "zh_text": zh_text,
                    "vi_text": vi_text,
                })

        if not self.instances:
            raise ValueError("Dataset empty after filtering; adjust max_len/min_len or data quality.")

    def _length_ok(self, tokens: Sequence[int]) -> bool:
        return self.config.min_len <= len(tokens) <= self.config.max_len

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        inst = self.instances[idx]
        return {
            "src": torch.tensor(inst["src"], dtype=torch.long),
            "tgt": torch.tensor(inst["tgt"], dtype=torch.long),
            "direction": inst["direction"],
            "zh_text": inst["zh_text"],
            "vi_text": inst["vi_text"],
        }


def build_collate_fn(pad_id: int):
    def _pad(seqs: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(seq.size(0) for seq in seqs)
        batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            batch[i, : seq.size(0)] = seq
        return batch

    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "src": _pad([item["src"] for item in batch]),
            "tgt": _pad([item["tgt"] for item in batch]),
            "direction": [item["direction"] for item in batch],
            "zh_text": [item["zh_text"] for item in batch],
            "vi_text": [item["vi_text"] for item in batch],
        }

    return collate


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


def encode_context(model: TransformerModel, src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    src_pad_mask = (src_ids == pad_id)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)
    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_ln(enc_out)
    return enc_out


def mean_pool(enc_out: torch.Tensor, ids: torch.Tensor, pad_id: int, special_ids: Sequence[int]) -> torch.Tensor:
    mask = (ids != pad_id)
    for special in special_ids:
        mask = mask & (ids != special)
    mask = mask.float()
    summed = (enc_out * mask.unsqueeze(-1)).sum(dim=1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return summed / denom


def contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, tau: float) -> torch.Tensor:
    sim = torch.matmul(z_a, z_b.transpose(0, 1)) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i = F.cross_entropy(sim, labels)
    loss_j = F.cross_entropy(sim.transpose(0, 1), labels)
    return 0.5 * (loss_i + loss_j)


def encode_directional_texts(
    sp_model: spm.SentencePieceProcessor,
    texts: Sequence[str],
    lang_token: str,
    pad_id: int,
    max_len: int,
) -> torch.Tensor:
    encoded: List[List[int]] = []
    for text in texts:
        stripped = text.strip()
        if stripped.startswith(lang_token):
            seq = sp_model.encode(stripped, out_type=int)
        else:
            seq = sp_model.encode(f"{lang_token} {stripped}", out_type=int)
        if not seq:
            seq = [pad_id]
        encoded.append(seq[: max_len])
    max_len_batch = max(len(seq) for seq in encoded)
    batch = torch.full((len(encoded), max_len_batch), pad_id, dtype=torch.long)
    for i, seq in enumerate(encoded):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def compute_crosslingual_loss(
    model: TransformerModel,
    projection: ProjectionHead,
    sp_model: spm.SentencePieceProcessor,
    zh_texts: Sequence[str],
    vi_texts: Sequence[str],
    config: PlanCLBiCrossConfig,
    pad_id: int,
    special_ids: Sequence[int],
) -> torch.Tensor:
    if len(zh_texts) < 2:
        return torch.zeros(1, device=config.device)

    ids_zh_vi = encode_directional_texts(sp_model, zh_texts, config.vi_token, pad_id, config.max_len).to(config.device)
    ids_vi_vi = encode_directional_texts(sp_model, vi_texts, config.vi_token, pad_id, config.max_len).to(config.device)
    ids_vi_zh = encode_directional_texts(sp_model, vi_texts, config.zh_token, pad_id, config.max_len).to(config.device)
    ids_zh_zh = encode_directional_texts(sp_model, zh_texts, config.zh_token, pad_id, config.max_len).to(config.device)

    enc_zh_vi = encode_context(model, ids_zh_vi, pad_id)
    enc_vi_vi = encode_context(model, ids_vi_vi, pad_id)
    enc_vi_zh = encode_context(model, ids_vi_zh, pad_id)
    enc_zh_zh = encode_context(model, ids_zh_zh, pad_id)

    z_zh_vi = projection(mean_pool(enc_zh_vi, ids_zh_vi, pad_id, special_ids))
    z_vi_vi = projection(mean_pool(enc_vi_vi, ids_vi_vi, pad_id, special_ids))
    z_vi_zh = projection(mean_pool(enc_vi_zh, ids_vi_zh, pad_id, special_ids))
    z_zh_zh = projection(mean_pool(enc_zh_zh, ids_zh_zh, pad_id, special_ids))

    cl_vi_space = contrastive_loss(z_zh_vi, z_vi_vi, config.contrastive_tau)
    cl_zh_space = contrastive_loss(z_vi_zh, z_zh_zh, config.contrastive_tau)
    return 0.5 * (cl_vi_space + cl_zh_space)


def greedy_decode_vi(
    model: TransformerModel,
    src_ids: torch.Tensor,
    sp_model: spm.SentencePieceProcessor,
    config: PlanCLBiCrossConfig,
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


def compute_ce_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    gold = targets[:, 1:]
    logits_flat = logits.reshape(-1, logits.size(-1))
    gold_flat = gold.reshape(-1)
    return criterion(logits_flat, gold_flat)


def train_epoch(
    model: TransformerModel,
    projection: ProjectionHead,
    sp_model: spm.SentencePieceProcessor,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupInverseSqrtScheduler,
    config: PlanCLBiCrossConfig,
    pad_id: int,
    special_ids: Sequence[int],
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
        zh_texts = batch["zh_text"]
        vi_texts = batch["vi_text"]

        optimizer.zero_grad()

        logits = model(src, tgt)
        ce_loss = compute_ce_loss(logits, tgt, criterion)

        if global_step < pretrain_steps:
            lambda_cross = 0.0
            cl_loss = torch.zeros(1, device=config.device)
        else:
            warmed = max(0, global_step - pretrain_steps)
            warmup_ratio = min(1.0, warmed / max(1, config.cross_warmup_steps))
            lambda_cross = config.cross_lambda_max * warmup_ratio
            cl_loss = compute_crosslingual_loss(
                model,
                projection,
                sp_model,
                zh_texts,
                vi_texts,
                config,
                pad_id,
                special_ids,
            )

        loss = ce_loss + lambda_cross * cl_loss
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
            "λ": f"{lambda_cross:.3f}",
            "lr": f"{scheduler.get_lr():.2e}",
        })

    steps = len(dataloader)
    return total_ce / steps, total_cl / steps, total_loss / steps, global_step


def evaluate(
    model: TransformerModel,
    projection: ProjectionHead,
    sp_model: spm.SentencePieceProcessor,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: PlanCLBiCrossConfig,
    pad_id: int,
    special_ids: Sequence[int],
    global_step: int,
    pretrain_steps: int,
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
            zh_texts = batch["zh_text"]
            vi_texts = batch["vi_text"]

            logits = model(src, tgt)
            ce_loss = compute_ce_loss(logits, tgt, criterion)
            total_ce += ce_loss.item()

            cl_loss = compute_crosslingual_loss(
                model,
                projection,
                sp_model,
                zh_texts,
                vi_texts,
                config,
                pad_id,
                special_ids,
            )
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
        lambda_cross = 0.0
    else:
        warmed = max(0, global_step - pretrain_steps)
        warmup_ratio = min(1.0, warmed / max(1, config.cross_warmup_steps))
        lambda_cross = config.cross_lambda_max * warmup_ratio

    metrics = {
        "val_ce": avg_ce,
        "val_cl": avg_cl,
        "val_lambda": lambda_cross,
        "val_total": avg_ce + lambda_cross * avg_cl,
    }

    if sacrebleu is not None and hyps and refs:
        metrics["bleu_zh2vi"] = sacrebleu.corpus_bleu(hyps, [refs]).score
    else:
        metrics["bleu_zh2vi"] = 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan-CL-BI-CROSS training")
    parser.add_argument("--train-src", default=None, help="Path to source training file")
    parser.add_argument("--train-tgt", default=None, help="Path to target training file")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--save-dir", default=None, help="Checkpoint directory override")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--tokenizer-prefix", default=None, help="Base prefix for tokenizer output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PlanCLBiCrossConfig()
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

    print("PLAN-CL-BI-CROSS training")
    print(f"Device: {config.device}")

    samples = load_interleaved_samples(config.train_src_file, config.train_tgt_file)
    print(f"Loaded {len(samples):,} bilingual samples")

    ensure_tokenizer(config, samples)

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f"{config.spm_prefix}.model")
    vocab_size = sp_model.GetPieceSize()
    print(f"SentencePiece vocab: {vocab_size}")
    tokenizer_payload = package_tokenizer(config.spm_prefix)

    train_samples, val_samples = split_samples(samples, config.val_ratio)
    print(f"Train samples: {len(train_samples):,} | Val samples: {len(val_samples):,}")

    train_dataset = PlanCLBiCrossDataset(train_samples, sp_model, config, include_dirs=("zh2vi", "vi2zh"))
    val_dataset = PlanCLBiCrossDataset(val_samples, sp_model, config, include_dirs=("zh2vi",))
    print(
        f"Filtered train instances: {len(train_dataset):,} (dropped {train_dataset.dropped}) | "
        f"Val instances: {len(val_dataset):,} (dropped {val_dataset.dropped})"
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

    model_params = sum(p.numel() for p in model.parameters())
    proj_params = sum(p.numel() for p in projection.parameters())
    print(
        f"Model params: {model_params:,} | Projection params: {proj_params:,} | "
        f"Combined: {model_params + proj_params:,}"
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
            sp_model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            pad_id,
            special_ids,
            global_step,
            pretrain_steps,
        )
        print(f"Train | CE: {ce_loss:.4f} | CL: {cl_loss:.4f} | Total: {total_loss:.4f}")

        metrics = evaluate(
            model,
            projection,
            sp_model,
            val_loader,
            criterion,
            config,
            pad_id,
            special_ids,
            global_step,
            pretrain_steps,
        )
        print(
            f"Val  | CE: {metrics['val_ce']:.4f} | CL: {metrics['val_cl']:.4f} | "
            f"λ: {metrics['val_lambda']:.4f} | Total: {metrics['val_total']:.4f} | "
            f"BLEU zh→vi: {metrics['bleu_zh2vi']:.2f}"
        )

        if epoch % config.save_every == 0:
            ckpt_path = os.path.join(config.save_dir, f"plan_cl_bi_cross_epoch_{epoch}.pt")
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
            best_path = os.path.join(config.save_dir, "plan_cl_bi_cross_best.pt")
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
