# Prompt Expansion with Multimodal Models for Enhanced Scene Rendering

This repository contains the complete final implementation repository for our paper:

**Prompt Expansion with Multimodal Models for Enhanced Scene Rendering**

This work focuses on improving text-to-image scene generation for complex, multi-clause, and semantically rich prompts through a two-stage pipeline:

1. **Prompt expansion** using a multimodal/LLM-based enrichment step to make scene descriptions more explicit.
2. **Targeted model adaptation** using **LoRA fine-tuning** on the **cross-attention layers of Stable Diffusion v1.4**.

This work bridge the gap between user intent and generated output by improving semantic grounding, preserving prompt attributes and relationships, and producing richer, more coherent scene renderings for prompts that standard text-to-image models often fail to capture faithfully.

Ultimately, the final image should better match the human user's intended meaning, visual expectation, and perceived quality, regardless of whether automated metrics alone appear strong or weak.

## Paper Overview

Although Stable Diffusion is capable of generating images from text prompts, the generated result often does not fully satisfy the user. In practice, there is frequently a gap between:

- what the user is actually imagining
- what the user writes in the prompt
- what the text-to-image model finally understands from that prompt

Our work is motivated by this gap between user intent and prompt-to-image generation quality.

Through experimentation, we observed that many unsatisfactory results are not only caused by model limitations, but also by the fact that user prompts often under-express the full richness of the intended scene. Important details related to atmosphere, composition, lighting, spatial relations, or semantic emphasis may remain implicit in the user's mind and therefore never reach the generation model clearly.

We also found that for domain-specific scenes, a general-purpose text-to-image model often does not fully satisfy user expectations. Even when the image is visually plausible, it may fail to capture the stylistic, semantic, or contextual richness expected in specialized domains.

To address this, the paper proposes a two-part solution:

- **Prompt expansion**, which enriches the user prompt into a more detailed and visually expressive prompt while preserving the original intent and properties
- **Domain-specific fine-tuning**, which improves grounding and visual quality for targeted scene categories

We fine-tuned the model across **10 domains**:

- Nature
- Architecture
- Sci-Fi
- Fantasy
- Historical
- Abstract
- Artistic
- Wildlife
- Everyday Life
- Emotions

Our experiments indicate that:

- general-purpose models often fall short of user satisfaction for domain-specific generation
- the fine-tuned model performs best for domain-specific generation
- expanded prompts produce richer and more semantically faithful image descriptions
- the combination of fine-tuning and prompt expansion leads to outputs that better match overall user expectations and satisfaction

In short, the paper argues that better image generation is not only a matter of a stronger backbone model, but also of narrowing the gap between user imagination and the final generation prompt seen by the model.

## Project Status

This repository currently contains:

- the paper implementation code
- the curated dataset metadata and caption splits used for training
- dataset reconstruction utilities
- the LoRA training script and configs
- the final fine-tuned model link
- the final inference and prompt-expansion generation scripts

This is the final complete project repository for the work, including the training pipeline, dataset organization, reconstruction utilities, and generation scripts.

## Model

The models used in this project are:

- Original base model: `CompVis/stable-diffusion-v1-4`
- Link: https://huggingface.co/CompVis/stable-diffusion-v1-4

- Fine-tuned model: `akshaysharma2277/textToImageFinetunedModelSD14`
- Link: https://huggingface.co/akshaysharma2277/textToImageFinetunedModelSD14

Note: The model link above is included from the project release information and may be updated further as the final model card is polished.

## Repository Contents

```text
.
|-- configs/
|   |-- train_lora_sd14.yaml
|   |-- train_lora_sd14_a10g.yaml
|   `-- train_lora_sd14_smoke.yaml
|-- dataset/
|   |-- captions/
|   |   |-- train.jsonl
|   |   `-- val.jsonl
|   `-- metadata/
|       |-- final_kept_dataset.csv
|       |-- final_kept_summary.csv
|       `-- validated_images.csv
|-- scripts/
|   |-- train_lora_sd14.py
|   |-- generate_phase1_images.py
|   |-- generate_phase2_images.py
|   |-- finalize_training_dataset.py
|   |-- regenerate_final_captions.py
|   |-- reconstruct_dataset_images.py
|   `-- reconstruct_frozen_dataset.py
|-- LICENSE
|-- README.md
|-- requirements.txt
```

## Method Overview

Our framework follows the paper design:

- **Base model**: `CompVis/stable-diffusion-v1-4`
- **Adaptation method**: LoRA
- **Trainable component**: U-Net only
- **LoRA target modules**: `to_q`, `to_k`, `to_v`, `to_out`
- **Text encoder**: frozen
- **VAE**: frozen
- **Additional supervision**: CLIP-based semantic alignment loss

The training objective combines:

- denoising loss (`MSE`)
- CLIP image-text cosine alignment loss

This lets the model preserve diffusion training behavior while improving prompt-image semantic consistency.

## Dataset

This repository includes the final curated training set metadata and split files used in the project as part of the complete implementation release.

### Final dataset size

- Total pairs: **1395**
- Train split: **1255**
- Validation split: **140**

### Sources used

- **COCO**
- **Unsplash**
- **WikiArt**
- curated source metadata from **PD12M**, **Re-LAION**, and **OpenBrush** during collection/finalization workflows

### Domain balance

The final kept dataset is organized across 10 domains:

- Nature
- Architecture
- Sci-Fi
- Fantasy
- Historical
- Abstract
- Artistic
- Wildlife
- Everyday Life
- Emotions

Per-domain counts are recorded in `dataset/metadata/final_kept_summary.csv`.

## What Is Included in This Repo

This repository includes:

- final caption splits in JSONL format
- final metadata CSVs
- dataset curation utilities
- reconstruction scripts for rebuilding the image folders

This repository does **not necessarily include all raw image assets directly in Git**, since some images are reconstructed from frozen metadata, cached exports, or external dataset sources.

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Depending on which scripts you use, you may also need:

- `huggingface_hub`
- `requests`
- `numpy`

## Training

The main training script is:

```bash
python scripts/train_lora_sd14.py --config configs/train_lora_sd14.yaml
```

### Available configs

- `configs/train_lora_sd14.yaml`: standard training config
- `configs/train_lora_sd14_a10g.yaml`: memory-aware config for A10G-style GPUs
- `configs/train_lora_sd14_smoke.yaml`: short smoke test config

### Training characteristics

- Stable Diffusion v1.4 backbone
- LoRA rank: `4`
- LoRA alpha: `4`
- 512x512 preprocessing
- CLIP regularization enabled
- mixed precision training (`fp16` when CUDA is available)

## Dataset Preparation Scripts

The repository includes the following dataset preparation and reconstruction scripts.

### 1. Finalize the training-ready dataset

Build the final kept dataset and split it into train/validation JSONL files:

```bash
python scripts/finalize_training_dataset.py
```

This script:

- merges accepted metadata sources
- selects target counts per domain
- creates final train/val JSONL caption files
- copies images into split folders when the source files are available locally

### 2. Regenerate final captions

```bash
python scripts/regenerate_final_captions.py
```

This script rewrites captions in `train.jsonl` and `val.jsonl` by joining final-kept rows back to upstream metadata sources.

### 3. Reconstruct dataset images

```bash
python scripts/reconstruct_dataset_images.py
```

This reconstructs `dataset/images/train` and `dataset/images/val` from:

- frozen metadata
- cached local exports
- source URLs when available

### 4. Reconstruct the frozen dataset more completely

```bash
python scripts/reconstruct_frozen_dataset.py --download-openbrush-shards
```

This script supports reconstruction of images from:

- PD12M
- Re-LAION
- OpenBrush shards on Hugging Face

## Current Data Files

Important committed data files include:

- `dataset/captions/train.jsonl`
- `dataset/captions/val.jsonl`
- `dataset/metadata/final_kept_dataset.csv`
- `dataset/metadata/final_kept_summary.csv`
- `dataset/metadata/validated_images.csv`

## Inference / Generation

This repository is the complete project repository and includes the assets needed for training, dataset preparation, reconstruction, and generation workflows.

The repository contains two generation scripts:

- **Phase 1**: image generation without prompt expansion
- **Phase 2**: image generation with DeepSeek-based prompt expansion followed by image generation for both the base prompt and expanded prompts

### Phase 1 Generation

Use Phase 1 when you want to generate images directly from the original user prompt.

```bash
python scripts/generate_phase1_images.py --model-type finetuned --prompt "A futuristic glass city under a blue sky"
```

Phase 1 supports both Hugging Face image-model options:

- base model via `--model-type base`
- fine-tuned project model via `--model-type finetuned`

Example:

```bash
python scripts/generate_phase1_images.py ^
  --model-type finetuned ^
  --prompt "A cinematic scene of a girl sitting on a chair with her tiger under golden light" ^
  --num-inference-steps 50 ^
  --width 512 ^
  --height 512 ^
  --num-images-per-prompt 2 ^
  --seed 42
```

### Phase 2 Generation

Use Phase 2 when you want to expand the prompt with DeepSeek and then generate images for:

- the original base prompt
- each expanded prompt variation

Example:

```bash
python scripts/generate_phase2_images.py ^
  --model-type finetuned ^
  --prompt "A cinematic scene of a girl sitting on a chair with her tiger under golden light" ^
  --num-inference-steps 50 ^
  --num-images-per-prompt 1 ^
  --num-variations 3 ^
  --seed 42
```

Phase 2 uses:

- a Hugging Face image model (`base` or `finetuned`)
- a Hugging Face DeepSeek model for prompt expansion

The generation scripts preserve the notebook-style parameters while making the workflow reproducible, scriptable, and project-ready.

### User-Tunable Parameters

Users can tune the generation behavior at run time through script arguments.

For Phase 1 and Phase 2 generation, the main tunable parameters include:

- model choice (`--model-type`)
- Hugging Face model ids (`--base-model-id`, `--finetuned-model-id`, `--deepseek-model-id`)
- input prompt (`--prompt`)
- negative prompt (`--negative-prompt`)
- number of inference steps (`--num-inference-steps`)
- image width and height (`--width`, `--height`)
- number of images per prompt (`--num-images-per-prompt`)
- seed (`--seed`)
- guidance scale (`--guidance-scale`)
- number of prompt expansions in Phase 2 (`--num-variations`)
- maximum expansion token budget in Phase 2 (`--max-expansion-tokens`)
- output directory and filename prefix (`--output-dir`, `--filename-prefix`)
- inference device (`--device`)

These parameters can be adjusted by the user depending on:

- desired image quality
- prompt complexity
- compute availability
- number of candidate generations required

### Code-Controlled Defaults

Some parts of the workflow are intentionally controlled inside the code and act as project defaults.

These include:

- the overall generation flow
- the prompt expansion instruction design
- output folder organization
- prompt parsing and cleanup logic
- fallback handling when expansion output is incomplete
- default sampling behavior for prompt expansion

These code-controlled defaults are chosen to provide a strong starting point, but advanced users may further refine them if they want to experiment with quality tuning or alternative prompt-expansion behavior.

## License

This repository is released under the license provided in `LICENSE`.

Please also respect the original licenses and usage restrictions of any upstream datasets or reconstructed assets.

## Citation

If you use this repository, dataset organization, or model weights, please cite the corresponding paper once the final citation block is ready.

## Acknowledgments

This work builds on top of:

- Stable Diffusion
- LoRA
- CLIP
- COCO
- Unsplash
- WikiArt
- PD12M
- Re-LAION
- OpenBrush

