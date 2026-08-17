# AutomatedTrainerCLI

Standalone **CLI-only** package for Swin Transformer V2 landmark (coordinate) regression on long-leg X-rays.

The model can be found under : https://drive.google.com/file/d/1xTS1mNibnGhPdLKTolgZkXawiNi9VAld/view?usp=sharing

No UI. Two scripts:

| Script | Purpose |
|--------|---------|
| `train.py` | Fine-tune from a checkpoint + data path |
| `infer.py` | Predict landmarks and save results to an output folder |

---

## Folder layout

```text
AutomatedTrainerCLI/
  train.py
  infer.py
  model_def_swin.py
  requirements.txt
  README.md
  .gitignore
  trainer/                 # training + inference library
  data/
    images/                # processed PNGs (created/filled by train)
    labels/                # processed *_points.txt
    unzipped/              # ZIP extract target
  weights/                 # put your .pth here; training writes here
  outputs/                 # inference results (suggested)
```

---

## 1. Setup

```bash
cd AutomatedTrainerCLI
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Use a CUDA build of PyTorch if you have a GPU.

Put your starting checkpoint here (example path):

```text
weights/best_model.pth
```

---

## 2. Quick start

### Fine-tune

```bash
python train.py --model weights/best_model.pth --data path/to/data.zip --output weights/my_run
```

### Inference (one image)

```bash
python infer.py --model weights/my_run/best_model.pth --input data/images/CASE.png --output outputs/CASE
```

### Inference (folder)

```bash
python infer.py --model weights/my_run/best_model.pth --input data/images --output outputs/batch
```

---

## 3. Retraining (`train.py`)

### Required

| Flag | Meaning |
|------|---------|
| `--model` | Checkpoint to fine-tune (`.pth` / `.pt`) |
| `--data` | ZIP, raw folder, or processed `images/`+`labels/` |

### Optional

| Flag | Default | Meaning |
|------|---------|---------|
| `--output` | `weights/train_run` | Checkpoint output directory |
| `--epochs` | `40` | Epochs |
| `--batch-size` | `1` | Batch size |
| `--lr` | `5e-5` | AdamW LR |
| `--val-split` | `0.15` | Validation fraction |
| `--num-workers` | `2` | DataLoader workers |
| `--seed` | `42` | Seed |
| `--image-ext` | `.png` | Processed image extension |
| `--non-strict-load` | off | Soft weight load |
| `--resume-optimizer` | off | Restore optimizer/scheduler from full `.pt` |

### What `--data` accepts

**A) ZIP**

```text
your_data.zip
├── Images/
│   └── <case_id>/
│       ├── <any>.png
│       └── <any>-1024.jpg
└── JsonVariables/
    └── <case_id>/
        └── object_*.json
```

Extracts to `data/unzipped/`, then writes:

- `data/images/<case_id>.png`
- `data/labels/<case_id>_points.txt`

**B) Raw unzipped folder** (same `Images/` + `JsonVariables/`)

```bash
python train.py --model weights/best_model.pth --data data/unzipped --output weights/run1
```

**C) Processed dataset**

```text
my_dataset/
  images/*.png
  labels/<stem>_points.txt
```

```bash
python train.py --model weights/best_model.pth --data my_dataset --output weights/run1
```

### Training outputs (`--output`)

| File | Contents |
|------|----------|
| `best_model.pth` | Plain `state_dict` — use with `infer.py` |
| `best.pt` | Full checkpoint (model + optimizer + meta) |
| `last.pt` | Full checkpoint for last epoch |

---

## 4. Inference (`infer.py`)

### Required

| Flag | Meaning |
|------|---------|
| `--model` | Checkpoint (prefer `best_model.pth`) |
| `--input` | Image file or folder |
| `--output` | Result directory |

### Optional

`--device cpu|cuda`, `--no-overlay`, `--no-json`, `--no-txt`

### Files written per image

For `CASE.png` → `--output outs/`:

```text
outs/
  CASE_points.txt      # name,x,y in ORIGINAL pixels
  CASE_points.json     # orig / model / normalized coords
  CASE_overlay.png     # drawn on 1024×512 canvas
```

---

## 5. Label format

Image: `data/images/<base>.png`  
Label: `data/labels/<base>_points.txt`

```text
tag,x,y
```

- Coordinates in **original image pixels**
- Names from `trainer/constants.py` (`LANDMARK_NAMES`)
- Missing names → `-1,-1` (masked in loss)
- **36** landmarks → head size **72**

```text
cmb31, cb31, cmt10, cb11, csp21, cmtc1, csp30, cmb10, cmtc0, cb10,
cmt20, csp10, cmt11, csp31, cmb11, cb30, csp11, cmb30, csp20, cmt21,
cmtc_mMPFA0, cmtc_mMPFA1,
labelBL0, labelBL1, labelBR0, labelBR1, labelTL0, labelTL1, labelTR0, labelTR1,
labelForCircleSetToBot0, labelForCircleSetToBot1,
labelBottomAngle0, labelBottomAngle1,
labelmMPFA0, labelmMPFA1
```

Raw Pointizr JSON keys are mapped in `BASE_TO_RIGHT` / `BASE_TO_LEFT` (smaller object ID = right/`0`, larger = left/`1`).

---

## 6. Model

| Setting | Value |
|---------|--------|
| Backbone | `swinv2_base_window8_256.ms_in1k` (timm) |
| Head | sigmoid → `[0, 1]` coords |
| Input | 3-channel (grayscale stacked) |
| Size | **1024 × 512** |
| Normalize | mean/std `0.5` |
| Loss | Masked Wing loss |
| Optimizer | AdamW + ReduceLROnPlateau |

See `model_def_swin.py` and `trainer/arch_presets.py`.

---

## 7. End-to-end example (Windows)

```bash
pip install -r requirements.txt

python train.py ^
  --model weights/best_model.pth ^
  --data C:\path\to\new_cases.zip ^
  --output weights\finetune_run ^
  --epochs 30

python infer.py ^
  --model weights\finetune_run\best_model.pth ^
  --input data\images ^
  --output outputs\batch
```

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| No image/label pairs | Check `*_points.txt` names; run with ZIP/raw so preprocess fills `data/` |
| Strict load fails | Checkpoint must match 36-landmark Swin head |
| OOM | Keep `--batch-size 1` |
| ZIP rejected | Need top-level `Images/` and `JsonVariables/` |
| Many `-1,-1` | Extend maps in `trainer/constants.py` |

---

## 9. Programmatic API

```python
from trainer.inference import run_inference, predict_image, build_inference_model

run_inference(
    model_path="weights/best_model.pth",
    input_path="data/images",
    output_dir="outputs/batch",
)

model, device = build_inference_model(checkpoint_path="weights/best_model.pth")
result = predict_image(model, "data/images/CASE.png", device=device)
print(result.points_orig["csp10"])
```
