# SF-TL54 xEdgeFace Project

This project adapts the pretrained EdgeFace model for cross-spectral face recognition on the SF-TL54 RGB-thermal dataset.

The workflow is:

1. Prepare or verify the processed dataset
2. Train the xEdgeFace adaptation
3. Evaluate the saved checkpoint
4. Compare the adapted model against the frozen EdgeFace baseline

## Project structure

- `sftl54_cross_spectral_pipeline.py` — builds paired RGB/thermal dataset and processed crops
- `sftl54_pytorch_dataset.py` — PyTorch dataset loader for SF-TL54
- `sftl54_edgeface_baseline.py` — baseline pretrained EdgeFace evaluation
- `sftl54_xedgeface_train.py` — xEdgeFace adaptation training script
- `sftl54_xedgeface_evaluate.py` — evaluation script for trained checkpoints
- `sftl54_transformer_adapter.py` — Phase 1 additive Stage-2 Transformer adapter
- `sftl54_experiment_1_train.py` — Phase 1 training script
- `sftl54_experiment_1_evaluate.py` — Phase 1 evaluation script
- `sftl54_processed/` — processed dataset crops and manifest
- `otroshi-edgeface-ce86851/` — local EdgeFace model code and pretrained checkpoints
- `sftl54_xedgeface_best.pt` — default trained checkpoint
- `sftl54_xedgeface_ln_st_s0_best.pt` — Stage-0 adapted model checkpoint

## Required environment

Use the workspace virtual environment:

```powershell
D:\RP\.venv311\Scripts\python.exe
```

You can also select this interpreter in VS Code via:

- `Ctrl+Shift+P`
- `Python: Select Interpreter`
- choose `D:\RP\.venv311\Scripts\python.exe`

## 1) Verify or prepare the dataset

The processed data already exists in this repo under `sftl54_processed/`, so the usual first check is:

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_pytorch_dataset.py"
```

This checks that the dataset loader works and the manifest is valid.

If the processed data is missing, rebuild it with:

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_cross_spectral_pipeline.py"
```

This script creates the cropped face dataset and Manifest CSV used by the training pipeline.

## 2) Run the baseline EdgeFace evaluation

This is the pretrained frozen baseline:

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_edgeface_baseline.py"
```

This gives the frozen EdgeFace RGB-to-thermal and thermal-to-RGB retrieval results without adaptation.

## 3) Train the adapted xEdgeFace model

### Recommended reproducible default run

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_train.py" `
    --batch-size 256 `
    --epochs 20 `
    --learning-rate 1e-4 `
    --lambda-distillation 0.75 `
    --margin 0 `
    --seed 42 `
    --output ".\sftl54_xedgeface_best.pt"
```

### Stage-0 adaptation run (paper-style ablation)

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_train.py" `
    --batch-size 256 `
    --epochs 20 `
    --learning-rate 1e-4 `
    --lambda-distillation 0.75 `
    --margin 0 `
    --include-stage0 `
    --seed 42 `
    --output ".\sftl54_xedgeface_ln_st_s0_best.pt"
```

If your machine is slow on CPU, you can temporarily reduce `--batch-size` for a quick smoke test, for example:

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_train.py" `
    --epochs 1 `
    --max-train-batches 1 `
    --max-val-batches 1 `
    --batch-size 8 `
    --seed 42
```

## 4) Evaluate a trained checkpoint

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_evaluate.py" `
    --checkpoint ".\sftl54_xedgeface_best.pt"
```

Or evaluate the Stage-0 model:

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_evaluate.py" `
    --checkpoint ".\sftl54_xedgeface_ln_st_s0_best.pt"
```

## 5) Experiment 1: Transformer Adapter

Experiment 1 keeps the EdgeFace-XS convolutional backbone and adds one
lightweight Transformer encoder after Stage 2. The verified Stage-2 tensor is
`[B, 100, 7, 7]`, which becomes 49 tokens of width 100. Mean-pooled
Transformer tokens are projected to 512 dimensions and added residually to the
original EdgeFace embedding. The projection is zero-initialized, so the
experiment starts from the original representation rather than replacing it.

Train the full Phase 1 experiment:

```powershell
cd D:\RP
& ".\.venv311\Scripts\python.exe" ".\sftl54_experiment_1_train.py" `
    --batch-size 256 `
    --epochs 20 `
    --learning-rate 1e-4 `
    --lambda-distillation 0.75 `
    --margin 0 `
    --seed 42 `
    --output ".\Experiment_1_Transformer_Adapter.pt"
```

Evaluate both RGB-to-thermal and thermal-to-RGB retrieval:

```powershell
& ".\.venv311\Scripts\python.exe" ".\sftl54_experiment_1_evaluate.py" `
    --checkpoint ".\Experiment_1_Transformer_Adapter.pt"
```

The CPU smoke test used one training and one validation batch only; its
metrics are diagnostic and must not be interpreted as the final experiment
result. A complete run is required before comparing accuracy against the
40.88%/68.32% RGB-to-thermal and 40.53%/68.67% thermal-to-RGB xEdgeFace
baseline.

## 6) Important interpretation notes

- The paper reports experimental settings such as `batch size = 256`, `learning rate = 1e-4`, `epochs = 20`, `lambda = 0.75`, and `margin = 0`.
- The project uses the SF-TL54 dataset and its own subject split, so results should be reported for this dataset specifically.
- The project is a strong adaptation of the xEdgeFace concept, and the training/evaluation pipeline is structured to support the paper-style ablation setup.
- Keep the results reproducible by using the fixed `--seed 42` value when reporting final RP outcomes.

## Typical command order

```powershell
cd D:\RP

# 1. dataset check
& ".\.venv311\Scripts\python.exe" ".\sftl54_pytorch_dataset.py"

# 2. baseline
& ".\.venv311\Scripts\python.exe" ".\sftl54_edgeface_baseline.py"

# 3. train adaptation
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_train.py" `
    --batch-size 256 `
    --epochs 20 `
    --learning-rate 1e-4 `
    --lambda-distillation 0.75 `
    --margin 0 `
    --include-stage0 `
    --seed 42 `
    --output ".\sftl54_xedgeface_ln_st_s0_best.pt"

# 4. evaluate
& ".\.venv311\Scripts\python.exe" ".\sftl54_xedgeface_evaluate.py" `
    --checkpoint ".\sftl54_xedgeface_ln_st_s0_best.pt"
```

## Notes

- The code assumes the project is executed from the workspace root.
- If any file is missing, make sure the local EdgeFace repo and dataset manifest are present.
- The training pipeline uses the local model checkpoint from `otroshi-edgeface-ce86851/checkpoints`.
