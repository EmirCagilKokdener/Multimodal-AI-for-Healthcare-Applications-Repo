#!/usr/bin/env python3
"""Generate Turkish radiology reports with MedGemma 1.5 + LoRA on CT-RATE slices.

Loads preprocessed .npz slices (from preprocess.py --split val) and the LoRA
adapter from the HuggingFace Hub by default. Writes one .txt per accession
and a merged results_all.xlsx.
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
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from prompt import PROMPT_TR

DEFAULT_LORA = "enderorman/medgemma-1.5-ct-rate-tr"
DEFAULT_MODEL = "google/medgemma-1.5-4b-it"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, required=True, help="Validation reports .xlsx")
    p.add_argument("--slices-dir", type=Path, required=True, help="Directory of .npz slices")
    p.add_argument("--out-dir", type=Path, required=True, help="Where per-scan .txt + results_all.xlsx are written")
    p.add_argument("--lora", default=DEFAULT_LORA,
                   help="LoRA adapter: HF repo ID or local path (default: %(default)s)")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    return p.parse_args()


def generate_report(model, processor, slice_images, prompt_text: str, max_new_tokens: int,
                    repetition_penalty: float, no_repeat_ngram_size: int) -> str:
    content = [{"type": "text", "text": prompt_text}]
    for i, im in enumerate(slice_images, 1):
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": f"SLICE {i}"})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, continue_final_message=False,
        return_tensors="pt", tokenize=True, return_dict=True,
    )

    with torch.inference_mode():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_cache=True,
        )

    out_text = processor.post_process_image_text_to_text(generated, skip_special_tokens=True)[0]
    in_text = processor.post_process_image_text_to_text(inputs["input_ids"], skip_special_tokens=True)[0]
    pos = out_text.find(in_text)
    if 0 <= pos <= 2:
        out_text = out_text[pos + len(in_text):].strip()
    return out_text


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_id, token=token, device_map="auto", quantization_config=bnb,
    )
    model = PeftModel.from_pretrained(base, args.lora, token=token)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id, token=token)
    print(f"  base:  {args.model_id}\n  lora:  {args.lora}\n")

    df = pd.read_excel(args.reports)
    df["AccessionNo"] = df["AccessionNo"].astype(str).str.strip()

    already = {f.name.replace("_report_tr.txt", "") for f in args.out_dir.glob("*_report_tr.txt")}
    remaining = [a for a in df["AccessionNo"] if a not in already]
    print(f"total={len(df)}  already_done={len(already)}  remaining={len(remaining)}\n")

    ok = fail = skipped = no_slice = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="generating"):
        acc = row["AccessionNo"]
        out_txt = args.out_dir / f"{acc}_report_tr.txt"
        if acc in already:
            skipped += 1
            continue

        npz_path = args.slices_dir / f"{acc}.npz"
        if not npz_path.exists():
            no_slice += 1
            (args.out_dir / f"{acc}_ERROR.txt").write_text(f"no preprocessed slices: {npz_path}")
            continue

        slice_images = None
        try:
            data = np.load(npz_path)
            slices_arr = data["slices"]
            slice_images = [Image.fromarray(slices_arr[i]) for i in range(slices_arr.shape[0])]
            del slices_arr, data

            report = generate_report(
                model, processor, slice_images, PROMPT_TR,
                args.max_new_tokens, args.repetition_penalty, args.no_repeat_ngram_size,
            )
            out_txt.write_text(report, encoding="utf-8")
            ok += 1

            gt = str(row.get("Findings_TR", ""))[:120]
            print(f"\n[{ok}] {acc}\n  GT:   {gt}...\n  Pred: {report[:120]}...")

        except Exception as e:
            fail += 1
            (args.out_dir / f"{acc}_ERROR.txt").write_text(repr(e))
            print(f"  [fail] {acc}: {e}")
        finally:
            del slice_images
            gc.collect()
            torch.cuda.empty_cache()

    print("\nmerging results into Excel...")
    rows = []
    for _, row in df.iterrows():
        acc = str(row["AccessionNo"]).strip()
        txt = args.out_dir / f"{acc}_report_tr.txt"
        generated = txt.read_text(encoding="utf-8").strip() if txt.exists() else "[NOT GENERATED]"
        rows.append({
            "AccessionNo": acc,
            "GT_Findings_TR": str(row.get("Findings_TR", "")),
            "GT_Impressions_TR": str(row.get("Impressions_TR", "")),
            "Generated_Report": generated,
        })
    excel_out = args.out_dir / "results_all.xlsx"
    pd.DataFrame(rows).to_excel(excel_out, index=False, engine="openpyxl")

    print(f"\ndone: ok={ok} fail={fail} skipped={skipped} no_slice={no_slice}")
    print(f"excel: {excel_out}")


if __name__ == "__main__":
    main()
