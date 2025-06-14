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

from examples.eval.datasets_utils import load_dataset_local
from examples.eval.metrics import wer


Example = tuple[torch.Tensor, int, str, str]


def _evaluate_dataset(
    model: Spiritlm,
    dataset: Iterable[Example],
    use_wandb: bool,
    writer: SummaryWriter | None,
    total_size: int | None = None,
) -> float:
    total = 0.0
    count = 0
    # Using inference_mode disables gradient calculations and speeds up inference.
    # Ensure the underlying HF model is in eval mode before generation.
    model.model.eval()
    with torch.inference_mode():
        for wav, sr, transcript, sample_id in tqdm(dataset, desc="Evaluating", total=total_size):
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.squeeze(0)
            wav = wav.to(dtype=torch.float32)
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
            wandb.log({
                "sample_wer": sample_wer,
                "pred": pred,
                "ground_truth": ref,
                "id": sample_id,
            })
        else:
            assert writer is not None
            writer.add_scalar("sample_wer", sample_wer, count)
            writer.add_text(
                "pred_ref",
                f"id: {sample_id}\npred: {pred}\nref: {ref}",
                global_step=count,
            )
    return total / max(1, count)


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")

    model_name = os.getenv("MODEL_NAME", "spirit-lm-expressive-7b")
    dataset_name = os.getenv("DATASET", "librispeech")
    subset = os.getenv("DATASET_SET", "test-clean")
    # data_root = os.getenv("DATASET_ROOT", "data")

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

    dataset, dataset_size = load_dataset_local(dataset_name, subset)
    model = Spiritlm(model_name)
    avg_wer = _evaluate_dataset(model, dataset, use_wandb, writer, dataset_size)

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
