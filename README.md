# Separate-and-Detect: Unified Drum Transcription and Stem Generation via Latent Diffusion

This is the official repository for: [Separate-and-Detect: Unified Drum Transcription and Stem Generation via Latent Diffusion](https://github.com/ddman1101/dssdtm.github.io).

Traditional automatic drum transcription (ADT) directly performs transcription from full-mixture music. This challenging %end-to-end 
task requires models to understand both the presence of drums and distinguish between different drum pieces. This study leverages advances in music source separation to propose a separation-then-transcription pipeline: a 5-stem multi-track drum separator using latent diffusion generates individual drum stems, after which per-stem onset detection yields class-wise pianorolls. The latent diffusion separator denoises in the compact VAE latent and renders audio with a vocoder. Additional onset/timbre auxiliaries guide the separator during training to encourage percussive-aware representations.

In MDB and ENST datasets, this pipeline competes with strong U-Net baseline (\emph{LarsNet}), showing class-specific gains while uniquely providing both accurate transcription and editable audio stems. Analysis also reveals that lower separation reconstruction error doesn't always result in higher transcription accuracy, motivating transcription-centric objectives in separation models. This work demonstrates latent diffusion-based separation offers a viable alternative to direct transcription, achieving competitive accuracy while enabling downstream audio editing applications.

# Installation

This project supports two conda environments for different use cases:

## Environment Setup

### Environment 1: `musicldm_env`
- **Purpose**: Original environment for MusicLDM related tasks
- **Usage**: For running the original batch evaluation scripts
- **Python**: 3.9
- **Includes**: librosa, madmom, scipy, numpy, matplotlib
- **Optimized for**: batch processing and evaluation

### Environment 2: `onset_detect`
- **Purpose**: Onset Detection environment
- **Usage**: For running the single audio onset detection script
- **Python**: 3.10
- **Includes**: librosa, madmom, scipy, numpy
- **Optimized for**: single audio onset detection

### Quick Setup

Create environments from yml files:
```bash
# Create musicldm_env (for batch evaluation)
conda env create -f musicldm_env.yml

# Create onset_detect environment (for single audio detection)
conda env create -f onset_detect.yml
```

Activate environments:
```bash
# For batch evaluation and training
conda activate musicldm_env

# For single audio onset detection
conda activate onset_detect
```


### (Be careful !) Fix madmom Import Issue

Modify the madmom package to fix compatibility issues:

1. Navigate to the madmom processors file:
   ```
   <conda_env_path>/lib/python3.10/site-packages/madmom/processors.py 
   ```

2. Edit line 23:

   **Change from:**
   ```python
   from collections import MutableSequence
   ```
   
   **Change to:**
   ```python
   from collections.abc import MutableSequence
   ```

### Usage Examples

**Batch Evaluation (musicldm_env):**
```bash
conda activate musicldm_env
python integrated_train.py --config <config_path>
```

**Single Audio Detection (onset_detect):**
```bash
conda activate onset_detect
bash run_single_onset_detection.sh
```


# Data

We use the StemGMD and IDMT-SMT-Drums datasets in this project.

Please download them from the following links and organize them to match the structure under the `data` folder in this repository:

- StemGMD: https://zenodo.org/records/7860223
- IDMT-SMT-Drums: https://zenodo.org/records/7544164

After downloading, preprocess and arrange your data to mirror the examples under the `data` directory (all examples reside in `data`). This ensures the training and evaluation scripts can locate audio and annotations correctly.

# Training MSG-LD

After data and conda environments are installed properly, you will need to download components of MusicLDM that are used for MSG-LD too. For this please 

```
# Download hifigan-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/hifigan-ckpt.ckpt

# Download vae-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/vae-ckpt.ckpt

```

After placing these files in your preferred directory and updating their paths in the corresponding config, run the following to train MSG-LD:

```
python train_musicldm.py --config config/MSG-LD/integrated_musicldm.yaml
```

# Config YAML

Common configs under `config/MSG-LD/` and when to use them:

- `integrated_musicldm.yaml`
  - Purpose: Baseline MSG-LD (latent diffusion separator) without auxiliaries.
  - Use when: You want a simple baseline to compare against auxiliary branches.

- `integrated_musicldm_onset.yaml`
  - Purpose: Adds an onset auxiliary branch to encourage percussion-aware separation.
  - Use when: You want better alignment of percussive cues without the timbre auxiliary.

- `integrated_musicldm_onset_timbre.yaml`
  - Purpose: Adds both onset and timbre auxiliary branches (recommended for drums).
  - Use when: You want best downstream drum transcription with editable stems.

- `inference_musicldm_mdb_inference.yaml`
  - Purpose: Inference/evaluation on the MDB-Drums dataset.
  - Use when: Running separation (and configured evaluation) on MDB splits.
  - Make sure: Dataset roots, checkpoint paths, and output dirs are correct.

- `inference_musicldm_enst_inference.yaml`
  - Purpose: Inference/evaluation on the ENST-Drums dataset.
  - Use when: Running separation (and configured evaluation) on ENST splits.
  - Make sure: Dataset roots, checkpoint paths, and output dirs are correct.

Example (training with onset+timbre):
```bash
CUDA_VISIBLE_DEVICES=0 \
python train_musicldm.py --config config/MSG-LD/integrated_musicldm_onset_timbre.yaml
```

# Inference

Two typical inference paths, mapped to the two environments.

1) Separation / dataset-level evaluation (musicldm_env)
```bash
conda activate musicldm_env

# MDB-Drums inference/eval (config controls dataset split/paths/checkpoints)
CUDA_VISIBLE_DEVICES=0 \
python train_musicldm.py --config config/MSG-LD/inference_musicldm_mdb_inference.yaml

# ENST-Drums inference/eval
CUDA_VISIBLE_DEVICES=0 \
python train_musicldm.py --config config/MSG-LD/inference_musicldm_enst_inference.yaml
```
Tips:
- Set `CUDA_VISIBLE_DEVICES` to select a GPU (optional).
- Verify paths inside the inference yaml(s): checkpoints, dataset roots, and output directories.

2) Single-audio onset transcription (onset_detect)
```bash
conda activate onset_detect

# Quick start (uses example path inside the script)
bash run_single_onset_detection.sh

# Or specify your own input/output
python single_audio_onset_detection.py \
  --input_audio /absolute/path/to/val_0/mix/YourSong.wav \
  --output_dir /absolute/path/to/transcription_results
```
Output transcripts are saved as `.txt`, one onset per line:
```
<timestamp_seconds>    <drum_type>
```

