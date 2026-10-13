# MRIxFields Harmonization

Code for cross-field brain MRI harmonization and translation for the MRIxFields2026 challenge. The project contains training, inference, evaluation, and Docker submission utilities for translating MRI volumes across magnetic field strengths such as 0.1T, 1.5T, 3T, 5T, and 7T.



## Project Structure

```text
.
+-- Baseline/                 # Training and inference code
|   +-- configs/              # YAML configs for Task 1, Task 2, and Task 3
|   +-- mrixfields/           # Model, dataset, loss, and utility modules
|   +-- scripts/              # Preprocessing, training, inference, segmentation
+-- Evaluation/               # Local metric evaluation scripts
+-- Submission/               # Submission packaging utilities
+-- Tutorial/                 # Example notebooks
+-- docker_v2/                # Optional Docker inference/submission package
+-- environment.yml           # Conda environment
+-- .env.example              # Example path configuration
```



## Requirements

- Python 3.10
- Conda or Miniconda
- NVIDIA GPU recommended
- CUDA-compatible PyTorch
- Git LFS for large checkpoints such as `.pth`

Install Git LFS before pushing model weights:

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.nii.gz"
git add .gitattributes
```

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate mf
```

The environment installs PyTorch, TensorFlow, MONAI-related dependencies where used, image processing tools, and the editable `Baseline/` package.

If your machine uses CUDA 11.8 instead of CUDA 12.1, edit `environment.yml` and replace `cu121` with `cu118` before creating the environment.

## Configure Paths

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
copy .env.example .env
```

Edit `.env` and set your local paths:

```text
DATA_DIR=/path/to/MRIxFields2026/dataset
PREPROCESSED_DIR=/path/to/preprocessed_cache
OUTPUT_DIR=/path/to/training_outputs
INFERENCE_DIR=/path/to/inference_outputs
SYNTHSEG_DIR=/path/to/SynthSeg
DEVICE=cuda:0
```

Expected dataset layout:

```text
$DATA_DIR/
+-- Training_retrospective/
+-- Training_prospective/
+-- Validating_prospective/
+-- Testing_prospective/
```

Each split should contain modality and field-strength folders, for example:

```text
Validating_prospective/
+-- T1W/
    +-- 0.1T/
    +-- 1.5T/
    +-- 3T/
    +-- 5T/
    +-- 7T/
```

## Preprocess Data

Preprocessing extracts 2D axial slices from 3D NIfTI volumes and stores them in `PREPROCESSED_DIR`.

Run from the `Baseline/` folder:

```bash
cd Baseline
python scripts/preprocess.py extract-slices --splits retro_train pro_train
```

For a smaller debug run:

```bash
python scripts/preprocess.py extract-slices --splits retro_train --fields 0.1T 7T --debug
```

## Train

Training is controlled by YAML files under `Baseline/configs/`.

### Task 3: Conditioned Swin-UNETR CUT Model

This is the main unified any-to-any field harmonization setup.

```bash
cd Baseline
python scripts/train.py \
  --config configs/task3/cut_swin_unetr/condD_full6_stage2gan100_fromscratch.yaml \
  --mode pro_pretrained \
  --device cuda
```

The config uses:

- Stage 1 retrospective pretraining
- Stage 2 prospective paired fine-tuning
- Conditioned generator using source field, target field, field delta, and modality
- Conditional discriminator using source field, target field, and modality

If you want to run only the retrospective pretraining stage:

```bash
python scripts/train.py \
  --config configs/task3/cut_swin_unetr/condD_full6_stage2gan100_fromscratch.yaml \
  --mode retro_scratch \
  --device cuda
```

If you already have a pretraining checkpoint and want to fine-tune from it:

```bash
python scripts/train.py \
  --config configs/task3/cut_swin_unetr/condD_full6_stage2gan100_fromscratch.yaml \
  --mode pro_pretrained \
  --pretrain_ckpt /path/to/checkpoint_epoch6.pth \
  --device cuda
```

Resume an interrupted run:

```bash
python scripts/train.py \
  --config configs/task3/cut_swin_unetr/condD_full6_stage2gan100_fromscratch.yaml \
  --mode pro_pretrained \
  --resume /path/to/latest_checkpoint.pth \
  --device cuda
```

### Task 1 Example

Example: translate 0.1T T1W to 7T T1W using CUT.

```bash
cd Baseline
python scripts/train.py \
  --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
  --mode retro_scratch \
  --device cuda
```

### Task 2 Example

Example: translate 0.1T T1W to 3T T1W using CycleGAN.

```bash
cd Baseline
python scripts/train.py \
  --config configs/task2/cyclegan/0.1T_to_3T_T1W.yaml \
  --mode retro_scratch \
  --device cuda
```

### Multi-GPU Training

```bash
cd Baseline
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/task3/cut_swin_unetr/condD_full6_stage2gan100_fromscratch.yaml \
  --mode pro_pretrained \
  --dist
```

## Run Inference

Run inference from the `Baseline/` folder.

Example Task 3 inference from 0.1T to 7T:

```bash
cd Baseline
python scripts/inference.py \
  --config configs/task3/cut_swin_unetr/condD_full6_stage2gan100_fromscratch.yaml \
  --checkpoint /path/to/checkpoint_best_validation.pth \
  --input_dir /path/to/Validating_prospective/T1W/0.1T \
  --output_dir /path/to/inference_outputs/T1W/0.1T_to_7T \
  --source_field 0.1T \
  --target_field 7T \
  --modality T1W \
  --device cuda
```

Task 1 example:

```bash
python scripts/inference.py \
  --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
  --checkpoint /path/to/checkpoint_epoch100.pth \
  --input_dir /path/to/Validating_prospective/T1W/0.1T \
  --output_dir /path/to/inference_outputs/task1_0.1T_to_7T_T1W \
  --device cuda
```

## Evaluate

Evaluation is run from the repository root.

### Task 3 Evaluation

Task 3 uses voxel-level metrics only: nRMSE, SSIM, and LPIPS.

```bash
python Evaluation/evaluate.py \
  --pred_dir /path/to/inference_outputs/T1W/0.1T_to_7T \
  --target_dir /path/to/Validating_prospective/T1W/7T \
  --metrics nrmse ssim lpips \
  --output_csv results_task3.csv \
  --device cuda
```

### Task 1 / Task 2 Voxel-Level Evaluation

```bash
python Evaluation/evaluate.py \
  --pred_dir /path/to/inference_outputs/task1_0.1T_to_7T_T1W \
  --target_dir /path/to/Validating_prospective/T1W/7T \
  --metrics nrmse ssim lpips \
  --output_csv results_voxel.csv \
  --device cuda
```

### Task 1 / Task 2 Full Evaluation With Segmentation

Dice and volume metrics require SynthSeg segmentations for both predictions and targets.

First install SynthSeg and set `SYNTHSEG_DIR` in `.env`.

Segment predictions:

```bash
python Evaluation/segment.py \
  --input_dir /path/to/inference_outputs/task1_0.1T_to_7T_T1W \
  --output_dir /path/to/inference_outputs/task1_0.1T_to_7T_T1W_seg
```

Segment targets:

```bash
python Evaluation/segment.py \
  --input_dir /path/to/Validating_prospective/T1W/7T \
  --output_dir /path/to/target_seg/T1W/7T
```

Run full evaluation:

```bash
python Evaluation/evaluate.py \
  --pred_dir /path/to/inference_outputs/task1_0.1T_to_7T_T1W \
  --target_dir /path/to/Validating_prospective/T1W/7T \
  --pred_seg_dir /path/to/inference_outputs/task1_0.1T_to_7T_T1W_seg \
  --target_seg_dir /path/to/target_seg/T1W/7T \
  --metrics nrmse ssim lpips dice volume \
  --output_csv results_full.csv \
  --device cuda
```

## Docker Inference Package

The `docker_v2/` folder is for challenge-style inference. It is not the main training pipeline.

Build the image:

```bash
cd docker_v2
docker build -t mrixfields-task3:v2 .
```

Run locally:

```bash
mkdir -p test_output
docker run --rm --gpus all \
  -v "$PWD/test_input:/input:ro" \
  -v "$PWD/test_output:/output" \
  mrixfields-task3:v2
```

On Windows PowerShell:

```powershell
mkdir test_output
docker run --rm --gpus all `
  -v "${PWD}\test_input:/input:ro" `
  -v "${PWD}\test_output:/output" `
  mrixfields-task3:v2
```

Expected success message:

```text
SUCCESS: wrote exactly N predictions.
```

## Uploading to GitHub

Large files must use Git LFS. Do not push `.pth` or `.nii.gz` files as normal Git objects.

```bash
git init
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.nii.gz"

git add .gitattributes
git add .
git commit -m "Add MRIxFields harmonization code"

git branch -M main
git remote add origin https://github.com/AramSalehi/mrixfields_harmonization.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin https://github.com/AramSalehi/mrixfields_harmonization.git
git push -u origin main
```

## Recommended `.gitignore`

```gitignore
__pycache__/
*.pyc
*.pyo
*.log
.env
.DS_Store

# Data and outputs
data/
datasets/
preprocessed/
outputs/
inference_outputs/
checkpoints/
runs/
wandb/
test_input/
test_output/

# Large archives
*.zip
*.rar
*.tar
*.tar.gz
```

## Notes

- Task 3 should use one unified conditional model for all field-to-field translations.
- Task 3 evaluation does not require SynthSeg segmentation.
- Task 1 and Task 2 full evaluation require SynthSeg for Dice and volume metrics.
- Keep private MRI data out of the public repository.
- Use GitHub Releases or Git LFS for model checkpoints.

## Citation

If you use this code in research, cite the MRIxFields2026 challenge and the relevant model papers used by the training configuration.
