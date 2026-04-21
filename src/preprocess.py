#!/usr/bin/env python3
"""Preprocess CT-RATE NIfTI volumes into slice tensors (.npz) for MedGemma training/inference.

For each accession in the reports Excel, downloads the NIfTI from HuggingFace,
extracts MAX_SLICES RGB slices at SLICE_SIZE resolution (3 CT windows per slice),
writes a compressed .npz, and deletes the NIfTI.
"""
import argparse
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "ibrahimhamamci/CT-RATE"
MAX_RETRIES = 5
RETRY_DELAYS = [10, 30, 60, 120, 300]


def build_ct_rate_subfolder(accession_no: str) -> str:
    parts = accession_no.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected AccessionNo format: {accession_no}")
    return f"dataset/{parts[0]}/{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}"


def get_nifti_path(accession_no: str, tmp_dir: Path, token: str) -> str:
    fname = accession_no.strip()
    if not fname.endswith(".nii.gz"):
        fname += ".nii.gz"
    clean_id = fname.replace(".nii.gz", "")
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        subfolder=build_ct_rate_subfolder(clean_id),
        filename=fname,
        local_dir=str(tmp_dir),
        token=token,
    )


def nifti_to_hu(path: str) -> np.ndarray:
    nii = nib.load(path)
    raw = np.asanyarray(nii.dataobj).astype(np.float32)
    hdr = nii.header
    slope = float(hdr.get("scl_slope", 1.0))
    inter = float(hdr.get("scl_inter", 0.0))
    if slope == 0 or np.isnan(slope):
        slope = 1.0
    if np.isnan(inter):
        inter = 0.0
    v = raw * slope + inter

    # Some CT-RATE volumes are stored with a +8192 offset; detect and correct.
    nz = v[v > 0]
    if nz.size:
        p50 = float(np.percentile(nz, 50))
        p99 = float(np.percentile(nz, 99))
        if p99 > 3000 and 5000 < p50 < 11000:
            v = v - 8192.0
    return v


def to_dhw_axial(vol: np.ndarray) -> np.ndarray:
    """Pick the axis whose orthogonal plane is closest to square & large — assumed axial."""
    assert vol.ndim == 3
    best_axis, best_score = 0, 1e18
    for axis in range(3):
        hw = [vol.shape[i] for i in range(3) if i != axis]
        score = abs(hw[0] - hw[1]) + (1024 - min(hw)) * 0.5
        if score < best_score:
            best_score, best_axis = score, axis
    return np.moveaxis(vol, best_axis, 0)


def _norm(ct2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip(ct2d, vmin, vmax).astype(np.float32)
    return ((x - vmin) / (vmax - vmin + 1e-6)) * 255.0


def window_rgb(ct2d_hu: np.ndarray) -> np.ndarray:
    window_clips = [(-1024, 1024), (-135, 215), (-1000, 400)]
    rgb = np.stack([_norm(ct2d_hu, a, b) for a, b in window_clips], axis=-1)
    return np.round(rgb, 0).astype(np.uint8)


def sample_slices(vol_dhw: np.ndarray, max_slices: int, out_size: int) -> np.ndarray:
    D = vol_dhw.shape[0]
    start, end = int(D * 0.08), int(D * 0.92)
    if end <= start:
        start, end = 0, D
    idxs = (
        np.linspace(start, end - 1, max_slices, dtype=int)
        if (end - start) > max_slices
        else np.arange(start, end, dtype=int)
    )
    slices = []
    for d in idxs:
        rgb = window_rgb(vol_dhw[d, :, :])
        rgb = cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
        slices.append(rgb)
    return np.stack(slices, axis=0)


def process_one(
    accession_no: str,
    tmp_dir: Path,
    out_dir: Path,
    max_slices: int,
    slice_size: int,
    token: str,
) -> bool:
    out_path = out_dir / f"{accession_no}.npz"
    if out_path.exists():
        return True

    for attempt in range(MAX_RETRIES):
        nifti_path = None
        try:
            nifti_path = get_nifti_path(accession_no, tmp_dir, token)
            vol = nifti_to_hu(nifti_path)
            vol = to_dhw_axial(vol)
            slices = sample_slices(vol, max_slices, slice_size)
            np.savez_compressed(out_path, slices=slices)
            del vol, slices
            gc.collect()
            return True
        except Exception as e:
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {accession_no}: {e} — {delay}s")
            time.sleep(delay)
        finally:
            if nifti_path and os.path.exists(nifti_path):
                try:
                    os.remove(nifti_path)
                except OSError:
                    pass
            gc.collect()
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--excel", type=Path, required=True, help="Reports Excel for this split")
    p.add_argument("--out-dir", type=Path, required=True, help="Where .npz files are written")
    p.add_argument("--tmp-dir", type=Path, required=True, help="Scratch dir for NIfTI downloads")
    p.add_argument("--max-slices", type=int, default=16)
    p.add_argument("--slice-size", type=int, default=384)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.excel)
    df["AccessionNo"] = df["AccessionNo"].astype(str).str.strip()
    all_acc = df["AccessionNo"].tolist()

    done = {f.stem for f in args.out_dir.glob("*.npz")}
    remaining = [a for a in all_acc if a not in done]

    print(f"split={args.split}  total={len(all_acc)}  done={len(done)}  remaining={len(remaining)}  workers={args.workers}")

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_one, acc, args.tmp_dir, args.out_dir,
                args.max_slices, args.slice_size, token,
            ): acc
            for acc in remaining
        }
        for fut in tqdm(as_completed(futures), total=len(remaining), desc="preprocessing"):
            acc = futures[fut]
            try:
                success = fut.result()
            except Exception as e:
                print(f"  [error] {acc}: {e}")
                success = False
            if success:
                ok += 1
            else:
                fail += 1
                with open(args.out_dir / "failed.txt", "a") as f:
                    f.write(f"{acc}\n")

    print(f"\ndone: ok={ok}  fail={fail}  elapsed={(time.time()-t0)/3600:.1f}h  out={args.out_dir}")


if __name__ == "__main__":
    main()
