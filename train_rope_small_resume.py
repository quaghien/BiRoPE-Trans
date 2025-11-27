#!/usr/bin/env python3
"""Resume training for the RoPE Transformer using an existing checkpoint tokenizer/model."""

import argparse
import os
import shutil
from datetime import datetime
from typing import Tuple

import torch

import train_rope_small as base


class Config(base.Config):
    """Placeholder so torch.load can resolve pickled __main__.Config objects."""
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume RoPE Transformer training from checkpoint")
    parser.add_argument("--checkpoint", default='/home/crl/hienhq/olym/checkpoints_v1_resume_v1/resume_epoch_45.pt', help="Path to checkpoint file containing model+tokenizer")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of additional epochs to run")
    parser.add_argument("--save-dir", type=str, default='checkpoints_resume_v2', help="Directory to store resumed checkpoints")
    parser.add_argument(
        "--tokenizer-prefix",
        type=str,
        default=None,
        help="Optional base directory/name to restore tokenizer files into",
    )
    parser.add_argument(
        "--vi2zh-ratio",
        type=float,
        default=None,
        help="Override fraction of vi→zh samples per epoch (0-1)",
    )
    return parser.parse_args()


def apply_checkpoint_config(config: base.Config, ckpt_cfg: object) -> None:
    """Copy matching attributes from checkpoint config into the fresh config."""
    if ckpt_cfg is None:
        return
    for key, value in vars(config).items():
        if hasattr(ckpt_cfg, key):
            setattr(config, key, getattr(ckpt_cfg, key))


def materialize_tokenizer(payload: dict, prefix_hint: str) -> Tuple[str, str]:
    if not payload:
        raise ValueError("Checkpoint lacks tokenizer payload.")
    model_bytes = payload.get("model_bytes")
    vocab_bytes = payload.get("vocab_bytes")
    if model_bytes is None or vocab_bytes is None:
        raise ValueError("Tokenizer payload missing model or vocab bytes.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(prefix_hint) or "."
    base_name = os.path.basename(prefix_hint) or "spm"
    restore_dir = os.path.join(base_dir, f"tokenizer_rope_resume_{timestamp}")
    os.makedirs(restore_dir, exist_ok=True)
    prefix = os.path.join(restore_dir, base_name)

    with open(f"{prefix}.model", "wb") as f:  # noqa: PTH123
        f.write(model_bytes)
    with open(f"{prefix}.vocab", "wb") as f:  # noqa: PTH123
        f.write(vocab_bytes)
    return prefix, restore_dir


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = base.Config()
    ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    apply_checkpoint_config(config, ckpt.get("config"))

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.save_dir:
        config.save_dir = args.save_dir
    if args.vi2zh_ratio is not None:
        config.vi2zh_epoch_ratio = args.vi2zh_ratio

    os.makedirs(config.save_dir, exist_ok=True)

    tokenizer_payload = ckpt.get("tokenizer")
    target_prefix = args.tokenizer_prefix or config.spm_prefix
    config.spm_prefix, tokenizer_dir = materialize_tokenizer(tokenizer_payload, target_prefix)
    print(f"Restored tokenizer to {config.spm_prefix}.[model|vocab]")

    sp_model = base.spm.SentencePieceProcessor()
    sp_model.Load(f"{config.spm_prefix}.model")
    vocab_size = sp_model.GetPieceSize()

    model = base.TransformerModel(config, vocab_size).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_base,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay,
    )
    scheduler = base.WarmupInverseSqrtScheduler(optimizer, config.warmup_steps, config.lr_base)
    criterion = base.LabelSmoothedCrossEntropyLoss(
        smoothing=config.label_smoothing,
        ignore_index=0,
    )

    # ===== Data prep (mirrors base script) =====
    src_lines = []
    tgt_lines = []
    with open(config.train_src_file, "r", encoding="utf-8") as fsrc, open(
        config.train_tgt_file, "r", encoding="utf-8"
    ) as ftgt:
        for src_line, tgt_line in zip(fsrc, ftgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line and tgt_line:
                src_lines.append(src_line)
                tgt_lines.append(tgt_line)

    split_idx = int(0.99999 * len(src_lines))
    if split_idx % 2 != 0:
        split_idx -= 1
    split_idx = max(split_idx, 0)
    train_src, train_tgt = src_lines[:split_idx], tgt_lines[:split_idx]
    valid_all_src, valid_all_tgt = src_lines[split_idx:], tgt_lines[split_idx:]
    valid_src, valid_tgt = [], []
    for i in range(0, len(valid_all_src), 2):
        if i < len(valid_all_src):
            valid_src.append(valid_all_src[i])
            valid_tgt.append(valid_all_tgt[i])

    train_dataset = base.BidirectionalTranslationDataset(train_src, train_tgt, sp_model, config, is_training=True)
    valid_dataset = base.BidirectionalTranslationDataset(valid_src, valid_tgt, sp_model, config, is_training=False)

    print(
        "Resume dataset | total={} zh→vi={} vi→zh={}".format(
            len(train_dataset), len(train_dataset.zh2vi_indices), len(train_dataset.vi2zh_indices)
        )
    )

    valid_loader = base.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=base.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    def build_train_loader(epoch_number: int):
        active = list(train_dataset.zh2vi_indices)
        vi_slice = base.select_vi2zh_window(train_dataset.vi2zh_indices, epoch_number, config.vi2zh_epoch_ratio)
        active.extend(vi_slice)
        sampler = base.SubsetRandomSampler(active)
        loader = base.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=base.collate_fn,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        return loader, len(vi_slice)

    start_epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    best_val_loss = metrics.get("val_loss", float("inf"))
    best_val_bleu = metrics.get("bleu_zh2vi", 0.0)

    tokenizer_payload = ckpt.get("tokenizer")

    for local_epoch in range(1, config.num_epochs + 1):
        global_epoch = start_epoch + local_epoch
        train_loader, vi_slice_len = build_train_loader(global_epoch)
        train_loss = base.train_epoch(model, train_loader, criterion, optimizer, scheduler, config, global_epoch)
        val_loss, val_bleu = base.evaluate(model, valid_loader, criterion, sp_model, config)

        print(f"\n[Resume] Epoch {global_epoch} (local {local_epoch}/{config.num_epochs})")
        print(
            f"  vi→zh coverage: {vi_slice_len}/{len(train_dataset.vi2zh_indices)} samples (~{100 * config.vi2zh_epoch_ratio:.1f}% per epoch)"
        )
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}")
        print(f"  Valid BLEU: {val_bleu:.2f}")
        print(f"  Learning Rate: {scheduler.get_lr():.6f}")

        if global_epoch % config.save_every == 0:
            path = os.path.join(config.save_dir, f"resume_epoch_{global_epoch}.pt")
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu": val_bleu,
                    "config": config,
                    "tokenizer": tokenizer_payload,
                },
                path,
            )
            print(f"  ✓ Resume checkpoint saved to {path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bleu = val_bleu
            best_path = os.path.join(config.save_dir, "resume_best.pt")
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu": val_bleu,
                    "config": config,
                    "tokenizer": tokenizer_payload,
                },
                best_path,
            )
            print(f"  ✓ New best resume checkpoint saved to {best_path}")

    print("\nResume training complete.")
    print(f"Best val loss: {best_val_loss:.4f} | BLEU: {best_val_bleu:.2f}")
    shutil.rmtree(tokenizer_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
