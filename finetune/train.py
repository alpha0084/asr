"""Fine-tune Whisper for Malayalam-audio -> English (speech translation).

Turnkey: point --data at a dataset.json (built by build_dataset.py) and run. Each
entry needs {audio, ref_en}. English `ref_en` is the training label; Whisper's
translate task is fine-tuned so ML audio -> EN text directly.

    python finetune/train.py --data finetune/data/dataset.json --out finetune/model \
        --base openai/whisper-large-v3 --src-lang ml --epochs 5 --lora

IMPORTANT — data shape: Whisper only sees <= 30s per example. Each training example's
audio must be a short clip (<= --max-seconds) with the English for THAT clip. Full-call
recordings (minutes long) with one call-level transcript are an EVAL set, not a train set
— they must be segmented into <=30s clips with aligned English first (see README). This
script trains correctly on clip-level data; on long calls it truncates to --max-seconds
(and warns), which is only meaningful once clips are short.
"""
import argparse, json, subprocess, warnings
from pathlib import Path
import numpy as np


def decode_16k(path: str) -> np.ndarray:
    """Decode any audio (mpeg/mp3/wav/...) to 16kHz mono float32 via ffmpeg."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", str(path), "-f", "f32le",
                          "-ac", "1", "-ar", "16000", "-"], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.json with [{audio, ref_en}]")
    ap.add_argument("--base", default="openai/whisper-large-v3")
    ap.add_argument("--out", default="finetune/model")
    ap.add_argument("--src-lang", default="ml")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--lora", action="store_true", help="parameter-efficient LoRA fine-tune")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    args = ap.parse_args()

    import torch, evaluate
    from datasets import Dataset
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from dataclasses import dataclass

    proc = WhisperProcessor.from_pretrained(args.base, language=args.src_lang, task="translate")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = args.src_lang
    model.generation_config.task = "translate"
    model.generation_config.forced_decoder_ids = None
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64,
                               target_modules=["q_proj", "v_proj"], lora_dropout=0.05))
        model.print_trainable_parameters()

    raw = [e for e in json.loads(Path(args.data).read_text()) if e.get("audio") and e.get("ref_en")]
    cap = int(args.max_seconds * 16000)
    longs = 0

    def prep(e):
        nonlocal longs
        a = decode_16k(e["audio"])
        if len(a) > cap:
            longs += 1; a = a[:cap]
        feat = proc.feature_extractor(a, sampling_rate=16000).input_features[0]
        labels = proc.tokenizer(e["ref_en"]).input_ids
        return {"input_features": feat, "labels": labels}

    ds = Dataset.from_list(raw).map(prep, remove_columns=Dataset.from_list(raw).column_names)
    if longs:
        warnings.warn(f"{longs}/{len(raw)} examples exceeded {args.max_seconds}s and were TRUNCATED. "
                      f"For real training, segment calls into <=30s clips with aligned English (see README).")
    split = ds.train_test_split(test_size=args.eval_frac, seed=0)

    @dataclass
    class Collator:
        def __call__(self, feats):
            batch = proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats],
                                               return_tensors="pt")
            lab = proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
            labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
            if (labels[:, 0] == proc.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    bleu = evaluate.load("sacrebleu")
    def metrics(p):
        pred = np.where(p.predictions != -100, p.predictions, proc.tokenizer.pad_token_id)
        lab = np.where(p.label_ids != -100, p.label_ids, proc.tokenizer.pad_token_id)
        hyp = proc.tokenizer.batch_decode(pred, skip_special_tokens=True)
        ref = proc.tokenizer.batch_decode(lab, skip_special_tokens=True)
        return {"bleu": bleu.compute(predictions=hyp, references=[[r] for r in ref])["score"]}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=0.1,
        gradient_checkpointing=True, fp16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True,
        generation_max_length=225, logging_steps=10, report_to=[], load_best_model_at_end=True,
        metric_for_best_model="bleu", greater_is_better=True)

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=split["train"],
                             eval_dataset=split["test"], data_collator=Collator(),
                             compute_metrics=metrics, processing_class=proc.feature_extractor)
    trainer.train()
    trainer.save_model(args.out)
    proc.save_pretrained(args.out)
    print(f"\nsaved fine-tuned model -> {args.out}")


if __name__ == "__main__":
    main()
