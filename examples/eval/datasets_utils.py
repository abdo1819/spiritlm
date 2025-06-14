from __future__ import annotations

from typing import Iterable, Tuple

import torch
from datasets import load_dataset,DownloadConfig


Example = Tuple[torch.Tensor, int, str, str]


dc = DownloadConfig(
    resume_download=True,   # keep the partial .incomplete file
    max_retries=20,         # keep retrying the same URL
    storage_options={"timeout": 3600}
)


def _load_librispeech_from_hf(subset: str) -> tuple[Iterable[Example], int]:
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
    # hf_dataset = load_dataset("mini_librispeech_asr", hf_config, split=hf_split,download_config=dc,trust_remote_code=True)
    hf_dataset = load_dataset("librispeech_asr", hf_config, split=hf_split, download_config=dc, trust_remote_code=True)

    def _iter() -> Iterable[Example]:
        for sample in hf_dataset:
            yield (
                torch.tensor(sample["audio"]["array"]),
                sample["audio"]["sampling_rate"],
                sample.get("text", ""),
                str(sample.get("id", sample.get("file", ""))),
            )

    return _iter(), len(hf_dataset)


def load_librispeech(subset: str) -> tuple[Iterable[Example], int]:
    return _load_librispeech_from_hf(subset)


def _load_gigaspeech_from_hf(subset: str) -> tuple[Iterable[Example], int]:
    print(f"loadnig speechcolab/gigaspeech , XS , split {subset}")
    hf_dataset = load_dataset("speechcolab/gigaspeech", "XS", split=subset)

    def _iter() -> Iterable[Example]:
        for sample in hf_dataset:
            yield (
                torch.tensor(sample["audio"]["array"]),
                sample["audio"]["sampling_rate"],
                sample["text"],
                str(sample.get("segment_id", sample.get("id", ""))),
            )

    return _iter(), len(hf_dataset)


def load_gigaspeech(subset: str) -> tuple[Iterable[Example], int]:
    return _load_gigaspeech_from_hf(subset)


def load_dataset_local(name: str, subset: str) -> tuple[Iterable[Example], int]:
    name = name.lower()
    if name in {"librispeech", "libri"}:
        return load_librispeech(subset)
    if name in {"gigaspeech", "giga"}:
        return load_gigaspeech(subset)
    raise ValueError(f"Unsupported dataset '{name}'")
