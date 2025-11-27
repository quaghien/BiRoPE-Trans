#!/usr/bin/env python3
"""Create bidirectional training files with target augmentation from evaluation CSVs."""

import argparse
import os
import re
from typing import List, Sequence, Tuple

import pandas as pd

CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
VIETNAMESE_CHAR_RE = re.compile(
    r"[ăâêôơưđĂÂÊÔƠƯĐàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]"
)


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR_RE.search(text))


def contains_vietnamese(text: str) -> bool:
    return bool(VIETNAMESE_CHAR_RE.search(text))


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def select_targets(
    row: pd.Series,
    base_col: str,
    sample_prefix: str,
    bleu_prefix: str,
    sample_count: int,
    min_bleu: float,
    max_bleu: float,
    enforce_no_chinese: bool,
    enforce_no_vietnamese: bool,
) -> List[str]:
    base = normalize_text(row.get(base_col, ""))
    if not base:
        return []

    candidates: List[Tuple[str, float]] = []
    for idx in range(1, sample_count + 1):
        sample_text = normalize_text(row.get(f"{sample_prefix}_{idx}", ""))
        bleu_value = row.get(f"{bleu_prefix}_{idx}")
        if not sample_text:
            continue
        if bleu_value is None or pd.isna(bleu_value):
            continue
        bleu_float = float(bleu_value)
        if not (min_bleu <= bleu_float <= max_bleu):
            continue
        if enforce_no_chinese and contains_chinese(sample_text):
            continue
        if enforce_no_vietnamese and contains_vietnamese(sample_text):
            continue
        candidates.append((sample_text, bleu_float))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = [text for text, _ in candidates[:2]]
    while len(selected) < 2:
        selected.append(base)

    return [base] + selected


def process_direction(
    df: pd.DataFrame,
    src_col: str,
    base_tgt_col: str,
    sample_prefix: str,
    bleu_prefix: str,
    enforce_no_chinese: bool,
    enforce_no_vietnamese: bool,
    sample_count: int,
    min_bleu: float,
    max_bleu: float,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        src_text = normalize_text(row.get(src_col, ""))
        if not src_text:
            continue

        targets = select_targets(
            row,
            base_col=base_tgt_col,
            sample_prefix=sample_prefix,
            bleu_prefix=bleu_prefix,
            sample_count=sample_count,
            min_bleu=min_bleu,
            max_bleu=max_bleu,
            enforce_no_chinese=enforce_no_chinese,
            enforce_no_vietnamese=enforce_no_vietnamese,
        )

        if not targets:
            continue

        for tgt in targets:
            pairs.append((src_text, tgt))

    return pairs


def write_lines(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment bidirectional training files from BLEU CSVs")
    parser.add_argument(
        "--vi2zh-csv",
        default="/home/crl/hienhq/olym/outputs/vi2zh_eval.csv",
        help="Path to vi->zh evaluation CSV",
    )
    parser.add_argument(
        "--zh2vi-csv",
        default="/home/crl/hienhq/olym/outputs/zh2vi_eval.csv",
        help="Path to zh->vi evaluation CSV",
    )
    parser.add_argument(
        "--src-out",
        default="/home/crl/hienhq/olym/outputs/src_train.txt",
        help="Destination file for source sentences",
    )
    parser.add_argument(
        "--tgt-out",
        default="/home/crl/hienhq/olym/outputs/target_train.txt",
        help="Destination file for target sentences",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of model sample columns to inspect per CSV",
    )
    parser.add_argument(
        "--min-bleu",
        type=float,
        default=35.0,
        help="Minimum BLEU threshold (inclusive) for augmentation candidates",
    )
    parser.add_argument(
        "--max-bleu",
        type=float,
        default=85.0,
        help="Maximum BLEU threshold (inclusive) for augmentation candidates",
    )
    return parser.parse_args()


def interleave_pairs(
    first_pairs: Sequence[Tuple[str, str]],
    second_pairs: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Alternate entries from two lists, starting with the first sequence."""
    result: List[Tuple[str, str]] = []
    i = j = 0
    turn_first = True

    while i < len(first_pairs) or j < len(second_pairs):
        if turn_first and i < len(first_pairs):
            result.append(first_pairs[i])
            i += 1
            turn_first = False
            continue
        if (not turn_first) and j < len(second_pairs):
            result.append(second_pairs[j])
            j += 1
            turn_first = True
            continue

        if i < len(first_pairs):
            result.extend(first_pairs[i:])
            break
        if j < len(second_pairs):
            result.extend(second_pairs[j:])
            break

    return result


def main() -> None:
    args = parse_args()

    vi2zh_df = pd.read_csv(args.vi2zh_csv)
    zh2vi_df = pd.read_csv(args.zh2vi_csv)

    vi2zh_pairs = process_direction(
        vi2zh_df,
        src_col="vi_org",
        base_tgt_col="zh_org",
        sample_prefix="zh_sample",
        bleu_prefix="bleu",
        enforce_no_chinese=False,
        enforce_no_vietnamese=True,
        sample_count=args.sample_count,
        min_bleu=args.min_bleu,
        max_bleu=args.max_bleu,
    )

    zh2vi_pairs = process_direction(
        zh2vi_df,
        src_col="zh_org",
        base_tgt_col="vi_org",
        sample_prefix="vi_sample",
        bleu_prefix="bleu",
        enforce_no_chinese=True,
        enforce_no_vietnamese=False,
        sample_count=args.sample_count,
        min_bleu=args.min_bleu,
        max_bleu=args.max_bleu,
    )

    ordered_pairs = interleave_pairs(zh2vi_pairs, vi2zh_pairs)

    src_entries_all = [src for src, _ in ordered_pairs]
    tgt_entries_all = [tgt for _, tgt in ordered_pairs]

    if len(src_entries_all) != len(tgt_entries_all):
        raise RuntimeError(
            f"Source/target length mismatch: {len(src_entries_all)} vs {len(tgt_entries_all)}"
        )

    write_lines(args.src_out, src_entries_all)
    write_lines(args.tgt_out, tgt_entries_all)

    print(
        f"Wrote {len(src_entries_all)} parallel samples: "
        f"{args.src_out} (src) / {args.tgt_out} (tgt)."
    )


if __name__ == "__main__":
    main()
