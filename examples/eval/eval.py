import os
from pathlib import Path
from typing import Iterable

import torchaudio
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import GenerationConfig

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb optional
    wandb = None
from torch.utils.tensorboard import SummaryWriter

from spiritlm.model.spiritlm_model import (
    Spiritlm,
    ContentType,
    GenerationInput,
    OutputModality,
)

from examples.eval.datasets import load_dataset
from examples.eval.metrics import wer


Example = tuple[torch.Tensor, int, str]


def _evaluate_dataset(model: Spiritlm, dataset: Iterable[Example], use_wandb: bool, writer: SummaryWriter | None) -> float:
    total = 0.0
    count = 0
    for wav, sr, transcript in tqdm(dataset, desc="Evaluating"):
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        outputs = model.generate(
            output_modality=OutputModality.TEXT,
            interleaved_inputs=[GenerationInput(content=wav, content_type=ContentType.SPEECH)],
            generation_config=GenerationConfig(do_sample=False, max_new_tokens=256),
        )
        pred = outputs[0].content.strip().lower()
        ref = transcript.lower().strip()
        sample_wer = wer(ref, pred)
        total += sample_wer
        count += 1
        if use_wandb:
            wandb.log({"sample_wer": sample_wer})
        else:
            assert writer is not None
            writer.add_scalar("sample_wer", sample_wer, count)
    return total / max(1, count)


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")

    model_name = os.getenv("MODEL_NAME", "spirit-lm-expressive-7b")
    dataset_name = os.getenv("DATASET", "librispeech")
    subset = os.getenv("DATASET_SET", "test-clean")
    data_root = os.getenv("DATASET_ROOT", "data")

    use_wandb = wandb is not None and os.getenv("WANDB_PROJECT")
    writer = None
    if use_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"{model_name}-{dataset_name}-{subset}",
        )
    else:
        writer = SummaryWriter(log_dir=f"runs/{dataset_name}_eval")

    dataset = load_dataset(dataset_name, data_root, subset)
    model = Spiritlm(model_name)
    avg_wer = _evaluate_dataset(model, dataset, use_wandb, writer)

    if use_wandb:
        wandb.summary["wer"] = avg_wer
    else:
        assert writer is not None
        writer.add_scalar("wer", avg_wer)
        writer.flush()
        writer.close()
    print(f"Average WER: {avg_wer:.4f}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
