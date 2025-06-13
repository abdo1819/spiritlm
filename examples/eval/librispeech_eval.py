import os
from pathlib import Path

import torchaudio
import torch
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from transformers import GenerationConfig

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb optional
    wandb = None
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

from spiritlm.model.spiritlm_model import (
    Spiritlm,
    ContentType,
    GenerationInput,
    OutputModality,
)


def wer(ref: str, hyp: str) -> float:
    """Compute word error rate between reference and hypothesis."""
    r = ref.split()
    h = hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / max(1, len(r))


def main():
    load_dotenv(Path(__file__).parent / ".env")

    model_name = os.getenv("MODEL_NAME", "spirit-lm-expressive-7b")
    data_root = os.getenv("LIBRISPEECH_ROOT", "data/LibriSpeech")
    subset = os.getenv("LIBRISPEECH_SET", "test-clean")

    use_wandb = wandb is not None and os.getenv("WANDB_PROJECT")
    writer = None
    if use_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"{model_name}-{subset}",
        )
    else:
        writer = SummaryWriter(log_dir="runs/librispeech_eval")

    subset_dir = Path(data_root) / subset
    if subset_dir.exists():
        dataset = LIBRISPEECH(data_root, url=subset, download=False)
    else:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as err:  # pragma: no cover - datasets optional
            raise RuntimeError(
                f"LibriSpeech subset '{subset}' not found at {subset_dir}. "
                "Install the 'datasets' package or download the dataset manually."
            ) from err

        def _hf_config_and_split(name: str) -> tuple[str, str]:
            if name.startswith("test-"):
                config = "clean" if "clean" in name else "other"
                return config, "test"
            if name.startswith("dev-"):
                config = "clean" if "clean" in name else "other"
                return config, "validation"
            if name.startswith("train-clean-100"):
                return "clean", "train.100"
            if name.startswith("train-clean-360"):
                return "clean", "train.360"
            if name.startswith("train-other-500"):
                return "other", "train.500"
            raise ValueError(f"Unsupported subset {name}")

        hf_config, hf_split = _hf_config_and_split(subset)
        hf_dataset = load_dataset("librispeech_asr", hf_config, split=hf_split)
        dataset = [
            (
                torch.tensor(sample["audio"]["array"]),
                sample["audio"]["sampling_rate"],
                sample.get("text", ""),
                0,
                0,
                0,
            )
            for sample in hf_dataset
        ]
    model = Spiritlm(model_name)

    total_wer = 0.0
    for wav, sr, _, _, _, transcript in tqdm(dataset, desc="Evaluating"):
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
        total_wer += sample_wer
        if use_wandb:
            wandb.log({"sample_wer": sample_wer})
        else:
            assert writer is not None
            writer.add_scalar("sample_wer", sample_wer)

    avg_wer = total_wer / len(dataset)
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
