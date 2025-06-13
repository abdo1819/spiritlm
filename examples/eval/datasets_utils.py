from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch
import torchaudio
from datasets import load_dataset  # type: ignore


Example = Tuple[torch.Tensor, int, str]


def _load_librispeech_from_hf(subset: str) -> Iterable[Example]:
    def _hf_config_and_split(name: str) -> tuple[str, str]:
        if name.startswith("test-"):
            config = "clean" if "clean" in name else "other"
            return config, "test"
        if name.startswith("dev-"):
            config = "clean" if "clean" in name else "other"
            return config, "test"
        if name.startswith("train-clean-100"):
            return "clean", "train.100"
        if name.startswith("train-clean-360"):
            return "clean", "train.360"
        if name.startswith("train-other-500"):
            return "other", "train.500"
        raise ValueError(f"Unsupported subset {name}")

    hf_config, hf_split = _hf_config_and_split(subset)
    print(f"loading librispeech_asr , config {hf_config} split {hf_split}")
    hf_dataset = load_dataset("librispeech_asr", hf_config, split=hf_split,streaming=True)
    for sample in hf_dataset:
        yield (
            torch.tensor(sample["audio"]["array"]),
            sample["audio"]["sampling_rate"],
            sample.get("text", ""),
        )


def load_librispeech(subset: str) -> Iterable[Example]:
    yield from _load_librispeech_from_hf(subset)


def _load_gigaspeech_from_hf(subset: str) -> Iterable[Example]:
    print(f"loadnig speechcolab/gigaspeech , XS , split {subset}")
    hf_dataset = load_dataset("speechcolab/gigaspeech", "XS", split=subset)
    for sample in hf_dataset:
        yield (
            torch.tensor(sample["audio"]["array"]),
            sample["audio"]["sampling_rate"],
            sample["text"],
        )


def load_gigaspeech(subset: str) -> Iterable[Example]:
    yield from _load_gigaspeech_from_hf(subset)


def load_dataset_local(name: str, subset: str) -> Iterable[Example]:
    name = name.lower()
    if name in {"librispeech", "libri"}:
        return load_librispeech(subset)
    if name in {"gigaspeech", "giga"}:
        return load_gigaspeech(subset)
    raise ValueError(f"Unsupported dataset '{name}'")
