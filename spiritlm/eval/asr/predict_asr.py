#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

"""ASR Evaluation on LibriSpeech.

Example usage:

cd {SPIRITLM ROOT FOLDER}
export PYTHONPATH=.

torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/asr/predict_asr.py \
    --subset test-clean --model spirit-lm-expressive-7b \
    --eval --write_pred ./librispeech_pred.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import Audio, load_dataset
from transformers import GenerationConfig, set_seed
from tqdm import tqdm
import wandb

from spiritlm.model.spiritlm_model import (
    ContentType,
    GenerationInput,
    InterleavedOutputs,
    OutputModality,
    Spiritlm,
)


def write_jsonl(path: str, predictions: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for idx, result_dict in predictions.items():
            record = {"id": idx, **result_dict}
            json_string = json.dumps(record)
            f.write(json_string + "\n")
    print(f"{path} written")


def word_error_rate(reference: str, hypothesis: str) -> float:
    r = reference.split()
    h = hypothesis.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
    return d[len(r)][len(h)] / max(len(r), 1)


def run(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_rank = int(os.environ.get("RANK", 0))
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)

    set_seed(args.seed)
    spiritlm_model = Spiritlm(args.model)
    wandb.init(
        project=args.wandb_project,
        name=args.run_name or f"{args.model}-{args.subset}",
        disable=world_rank != 0,
    )

    def parse_subset(name: str) -> Tuple[str, str]:
        name = name.lower()
        mapping = {
            "test-clean": ("clean", "test"),
            "test-other": ("other", "test"),
            "dev-clean": ("clean", "validation"),
            "dev-other": ("other", "validation"),
        }
        if name not in mapping:
            raise ValueError(f"Unsupported subset {name}")
        return mapping[name]

    config, split = parse_subset(args.subset)

    dataset = load_dataset(
        "librispeech_asr",
        config,
        split=split,
        cache_dir=args.dataset_root,
        trust_remote_code=True,
    ).cast_column("audio", Audio(decode=True))

    dataset = dataset.shard(num_shards=world_size, index=world_rank)

    dataset_len = len(dataset)
    predictions = {}
    wers: List[float] = []
    for i, sample in enumerate(
        tqdm(
            dataset,
            total=dataset_len,
            disable=world_rank != 0,
            desc=f"Rank {world_rank} predict {args.subset}",
        )
    ):
        wav = sample["audio"]["array"]
        transcript = sample["text"]
        out: InterleavedOutputs = spiritlm_model.generate(
            output_modality=OutputModality.TEXT,
            interleaved_inputs=[
                GenerationInput(content=wav, content_type=ContentType.SPEECH)
            ],
            generation_config=GenerationConfig(
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=300,
                do_sample=True,
            ),
        )
        assert len(out) == 1
        predicted_text = out[0].content.strip()
        ref = transcript.lower().strip()
        wer = word_error_rate(ref, predicted_text.lower())
        wers.append(wer)
        predictions[str(i)] = {"pred": predicted_text, "ref": ref}
        wandb.log({"wer": wer}, step=i)

    if args.eval:
        gathered_predictions = [None for _ in range(world_size)]
        gathered_wers = [None for _ in range(world_size)]
        dist.gather_object(predictions, gathered_predictions if world_rank == 0 else None, dst=0)
        dist.gather_object(wers, gathered_wers if world_rank == 0 else None, dst=0)
        if world_rank == 0:
            all_wers = [w for ws in gathered_wers for w in ws]
            avg_wer = sum(all_wers) / len(all_wers)
            print(f"WER: {avg_wer * 100:.2f}% for subset {args.subset}")
            wandb.log({"avg_wer": avg_wer})
            all_predictions = {k: v for d in gathered_predictions for k, v in d.items()}
    else:
        all_predictions = predictions

    if args.write_pred is not None and world_rank == 0:
        write_jsonl(args.write_pred, all_predictions)

    wandb.finish()


def setup_env():
    os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/hf_cache",
        help="Local cache directory for the dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test-clean",
        help="Subset to evaluate (e.g. test-clean)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spirit-lm-expressive-7b",
        help="Model name or path",
    )
    parser.add_argument(
        "--write_pred",
        type=str,
        default=None,
        help="Path to save the predictions output",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="spiritlm-asr-eval",
        help="WandB project for logging",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional WandB run name",
    )
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    setup_env()
    run(args)
