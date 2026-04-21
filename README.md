# MedGemma 1.5 — CT-RATE (Turkish) Fine-tuning

Fine-tune and run inference with Google **MedGemma 1.5 4B IT** on the
[CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset to
generate **Turkish radiology reports** (`Bulgular` / `İzlenim`) from 3D chest CT.

The trained LoRA adapter is published at
**[enderorman/medgemma-1.5-ct-rate-tr](https://huggingface.co/enderorman/medgemma-1.5-ct-rate-tr)**
and is used by default in `inference.py` — no local checkpoint required to run inference.

## Layout

```
medgemma-ct-rate/
├── src/
│   ├── preprocess.py   # NIfTI → 16 RGB slices (.npz)
│   ├── train.py        # QLoRA fine-tune
│   ├── inference.py    # generate reports from .npz slices
│   └── prompt.py       # Turkish radiology prompt
├── scripts/            # thin shell wrappers that read env vars
│   ├── preprocess.sh
│   ├── train.sh
│   └── inference.sh
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

Requires a GPU with ≥16 GB VRAM (tested on a single A100 / 4-bit QLoRA).

```bash
git clone <your-fork-url>
cd medgemma-ct-rate

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set HF_TOKEN=hf_...
export $(grep -v '^#' .env | xargs)   # or: source .env after adding `export `
```

`HF_TOKEN` must have read access to:
- `google/medgemma-1.5-4b-it` (gated — request access first)
- `ibrahimhamamci/CT-RATE` (gated)
- `enderorman/medgemma-1.5-ct-rate-tr` (public)

## Data layout

Put the CT-RATE reports Excels in `./data/` (or set `DATA_ROOT`):

```
data/
├── train_reports_tr.xlsx
└── validation_reports_tr.xlsx
```

Slice caches are created automatically by `preprocess.sh` into
`data/slices/` (train) and `data/val_slices/` (val).

## Usage

### 1. Preprocess slices (runs for hours — tmux recommended)

```bash
./scripts/preprocess.sh train    # ~330 GB for 47k scans
./scripts/preprocess.sh val
```

Each NIfTI is downloaded, 16 RGB slices (384×384, 3 CT windows) are extracted,
saved as a compressed `.npz` (~7 MB), and the original NIfTI is deleted.

### 2. Fine-tune

```bash
./scripts/train.sh
```

Defaults: 3 epochs, batch 4, grad-accum 4, LR 2e-4, QLoRA r=16 on all
attention + MLP projections. Checkpoints go to `./checkpoints/`. Training
resumes automatically from the latest checkpoint if interrupted.

Override any hyperparameter via env vars — see `.env.example`.

### 3. Inference

```bash
./scripts/inference.sh
```

Loads the LoRA from HF Hub by default. To use a local checkpoint instead:

```bash
LORA=./checkpoints/checkpoint-8841 ./scripts/inference.sh
```

One `<AccessionNo>_report_tr.txt` per scan is written to `./results/`, plus a
merged `results_all.xlsx` with ground-truth vs. generated columns.

## Direct Python usage (no shell wrapper)

```bash
python src/preprocess.py --split train \
    --excel data/train_reports_tr.xlsx \
    --out-dir data/slices \
    --tmp-dir tmp_dl

python src/train.py \
    --reports data/train_reports_tr.xlsx \
    --slices-dir data/slices \
    --out-dir checkpoints

python src/inference.py \
    --reports data/validation_reports_tr.xlsx \
    --slices-dir data/val_slices \
    --out-dir results
```

## Model card

See the adapter's [HuggingFace page](https://huggingface.co/enderorman/medgemma-1.5-ct-rate-tr)
for training details, metrics, and intended use.

## License

Code: MIT.
Base model: subject to [MedGemma Health AI Developer Foundation terms](https://developers.google.com/health-ai-developer-foundations/terms).
Dataset: subject to the [CT-RATE license](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## Acknowledgements

- Base model: Google [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it)
- Dataset: [CT-RATE (Hamamci et al.)](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
