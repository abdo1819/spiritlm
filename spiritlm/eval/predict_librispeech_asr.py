import argparse
import json
from typing import List

from datasets import Audio, load_dataset
from jiwer import wer
from tqdm import tqdm
from transformers import GenerationConfig, set_seed

from spiritlm.model.spiritlm_model import (
    ContentType,
    GenerationInput,
    OutputModality,
    Spiritlm,
)


def transcribe(model: Spiritlm, audio_array) -> str:
    out = model.generate(
        output_modality=OutputModality.TEXT,
        interleaved_inputs=[GenerationInput(content=audio_array, content_type=ContentType.SPEECH)],
        generation_config=GenerationConfig(
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=100,
            do_sample=True,
        ),
    )
    assert len(out) == 1
    return out[0].content.strip()


def run(args: argparse.Namespace):
    set_seed(args.seed)
    ds = load_dataset(
        "librispeech_asr",
        args.config,
        split=args.split,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    model = Spiritlm(args.model)

    references: List[str] = []
    predictions: List[str] = []
    for sample in tqdm(ds, desc=f"Transcribing {args.config}/{args.split}"):
        references.append(sample["text"].strip())
        pred = transcribe(model, sample["audio"]["array"])
        predictions.append(pred)

    error = wer(references, predictions) * 100
    print(f"WER for {args.config}/{args.split}: {error:.2f}%")

    if args.write_pred:
        with open(args.write_pred, "w") as f:
            for sample, pred in zip(ds, predictions):
                f.write(json.dumps({"id": sample["id"], "pred": pred}) + "\n")
        print(f"Predictions saved to {args.write_pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["clean", "other"],
        default="clean",
        help="LibriSpeech configuration",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spirit-lm-expressive-7b",
        help="Model name or path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--write_pred",
        type=str,
        default=None,
        help="Optional path to write predictions as JSONL",
    )
    args = parser.parse_args()
    run(args)
