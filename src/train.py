#!/usr/bin/env python3
"""Fine-tune MedGemma 1.5 on CT-RATE (Turkish) with 4-bit QLoRA.

Expects slices/.npz files preprocessed by preprocess.py --split train and a
reports Excel with AccessionNo, Findings_TR, Impressions_TR columns.
"""
import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from prompt import PROMPT_TR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, required=True, help="Train reports .xlsx")
    p.add_argument("--slices-dir", type=Path, required=True, help="Directory of .npz slice files")
    p.add_argument("--out-dir", type=Path, required=True, help="Where checkpoints are saved")
    p.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=5)
    return p.parse_args()


def build_target_text(row) -> str | None:
    findings = str(row.get("Findings_TR", "") or "").strip()
    impression = str(row.get("Impressions_TR", "") or "").strip()
    if not findings and not impression:
        return None
    out = ""
    if findings:
        out += f"Bulgular:\n{findings}\n\n"
    if impression:
        out += f"İzlenim:\n{impression}"
    return out.strip()


def build_messages(images, target_text: str):
    content = [{"type": "text", "text": PROMPT_TR}]
    for i, im in enumerate(images, 1):
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": f"SLICE {i}"})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]


def make_collate_fn(processor, slices_dir: Path):
    def process_one(ex):
        data = np.load(slices_dir / f"{ex['accession_no']}.npz")
        slices_arr = data["slices"]  # (N, H, W, 3) uint8
        imgs = [Image.fromarray(slices_arr[i]) for i in range(slices_arr.shape[0])]

        x_full = processor.apply_chat_template(
            build_messages(imgs, ex["target"]),
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        x_prefix = processor.apply_chat_template(
            build_messages(imgs, ""),
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        prefix_len = x_prefix["input_ids"].shape[1]
        del slices_arr, data, imgs
        gc.collect()
        return x_full, prefix_len

    def collate(batch):
        all_full, prefix_lens = [], []
        for ex in batch:
            x_full, pl = process_one(ex)
            all_full.append(x_full)
            prefix_lens.append(pl)

        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"][0] for x in all_full], batch_first=True, padding_value=pad_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"][0] for x in all_full], batch_first=True, padding_value=0,
        )
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
            [x["token_type_ids"][0] for x in all_full], batch_first=True, padding_value=0,
        )

        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        for k in all_full[0].keys():
            if k in out:
                continue
            try:
                out[k] = torch.cat([x[k] for x in all_full], dim=0)
            except Exception as e:
                print(f"[warn] skipping key {k}: {e}")
        return out

    return collate


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.reports)
    df["AccessionNo"] = df["AccessionNo"].astype(str).str.strip()
    samples = []
    for _, row in df.iterrows():
        target = build_target_text(row)
        if target is None:
            continue
        acc = row["AccessionNo"]
        if not (args.slices_dir / f"{acc}.npz").exists():
            continue
        samples.append({"id": acc, "accession_no": acc, "target": target})

    ds = Dataset.from_list(samples)
    print(ds)
    print(ds[0])

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, token=token, device_map="auto", quantization_config=bnb,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, token=token)

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=make_collate_fn(processor, args.slices_dir),
    )

    last_ckpt = get_last_checkpoint(str(args.out_dir))
    if last_ckpt:
        print(f"resuming from {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("starting fresh")
        trainer.train()

    trainer.save_model(str(args.out_dir))
    print(f"saved LoRA to: {args.out_dir}  step={trainer.state.global_step}")


if __name__ == "__main__":
    main()
