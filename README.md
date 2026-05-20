# PanEchoTest

Run [PanEcho](https://github.com/CarDS-Yale/PanEcho) (Holste et al., JAMA 2025) on the full EchoNet-Dynamic dataset. Mac-compatible — works on Apple Silicon (MPS) and Intel (CPU).

---

## What this does

1. You download the 10,030-video EchoNet-Dynamic dataset from Stanford AIMI.
2. `run_panecho.py` loads the pretrained PanEcho model via PyTorch Hub and runs inference on every video, writing predictions to a CSV.

---

## 1 · Prerequisites

- macOS (Apple Silicon M1/M2/M3/M4 or Intel)
- Python 3.10+ **or** Conda / Miniconda
- ~50 GB free disk space for the dataset
- Internet access (to download model weights on first run, ~300 MB)

---

## 2 · Installation

### Option A — conda (recommended)

```bash
bash setup_mac.sh          # creates the 'panecho' conda env
conda activate panecho
```

### Option B — pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3 · Download EchoNet-Dynamic

The dataset requires a one-time registration with Stanford AIMI. There is no
automated download — follow these steps:

1. Go to **https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a**
2. Create a free account and agree to the Research Use Agreement.
3. Download **EchoNet-Dynamic.zip** (~7 GB compressed → ~28 GB unzipped).
4. Unzip it:

   ```bash
   unzip EchoNet-Dynamic.zip -d ~/data/
   ```

The resulting directory must contain:

```
EchoNet-Dynamic/
├── FileList.csv          # metadata + EF ground truth for all 10 030 videos
├── VolumeTracings.csv    # frame-level tracings
└── Videos/
    ├── 0X1A2B3C....avi
    └── ... (10 030 AVI files)
```

---

## 4 · Run PanEcho

```bash
python run_panecho.py --data_dir ~/data/EchoNet-Dynamic
```

The script:
- **auto-detects** the best device (MPS on Apple Silicon, CPU on Intel)
- streams results row-by-row to `panecho_results.csv` so you never lose progress
- **resumes automatically** if interrupted — just re-run the same command

### Common options

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Path to EchoNet-Dynamic root |
| `--output` | `panecho_results.csv` | Output CSV path |
| `--clip_len` | `16` | Frames sampled per video |
| `--split` | all | `train` / `val` / `test` only |
| `--max_videos` | all | Cap at N videos (for testing) |
| `--device` | auto | `mps` / `cpu` / `cuda` |
| `--overwrite` | false | Overwrite output instead of resuming |

### Quick smoke test (5 videos, CPU)

```bash
python run_panecho.py \
    --data_dir ~/data/EchoNet-Dynamic \
    --max_videos 5 \
    --device cpu \
    --output test_run.csv
```

---

## 5 · Output format

`panecho_results.csv` has one row per video:

| Column | Description |
|---|---|
| `FileName` | Video filename (without path) |
| `GT_EF` | Ground-truth ejection fraction |
| `GT_ESV` | Ground-truth end-systolic volume |
| `GT_EDV` | Ground-truth end-diastolic volume |
| `GT_Split` | Dataset split (TRAIN / VAL / TEST) |
| `EF` | PanEcho predicted EF (regression) |
| `LVSystolicFunction_cls0/1/2` | LV systolic function probabilities |
| *(39+ additional task columns)* | Classification probabilities and regression estimates |

---

## 6 · Device notes

| Mac type | Device used | Notes |
|---|---|---|
| Apple Silicon (M-series) | `mps` | ~3–5x faster than CPU; auto-selected |
| Intel Mac | `cpu` | Slower but fully supported |
| Any | `cuda` | If you have an NVIDIA eGPU |

MPS acceleration requires PyTorch >= 2.1. If you encounter an MPS-related
`RuntimeError`, the script falls back to CPU for that video automatically.

**Estimated runtime (10 030 videos):**
- Apple Silicon M2/M3 (MPS): ~45-90 minutes
- Intel Mac (CPU): ~3-6 hours

---

## Citation

```bibtex
@article{holste2025panecho,
  title   = {Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning},
  author  = {Holste, Gregory and Oikonomou, Evangelos K. and Tokodi, M{\'a}rton
             and Kov{\'a}cs, Attila and Wang, Zhangyang and Khera, Rohan},
  journal = {JAMA},
  year    = {2025}
}
```
