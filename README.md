# Vision–Language–Action Policy for Urban Driving

## Overview

This project implements a simple **Vision–Language–Action (VLA)** pipeline for high-level urban driving decisions.

Given:
- a **front camera RGB image**, and  
- a **natural-language description** of the scene,

The model predicts one of the high-level actions:

`MAINTAIN_SPEED`, `SLOW_DOWN`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`

and generates a short text explanation.

The system uses:
- **MobileNetV2** (ImageNet pretrained, frozen) as a vision encoder  
- a **keyword-based command encoder** for text  
- a small **MLP policy head**  
- a **template-based explanation module**

## Files

- `model_vla.py` – VLA model, action list, text parser, explanation module  
- `train_vla.py` – dataset loading, preprocessing, training loop, checkpoint saving  
- `inference_vla.py` – run inference on a single image + description

## Training (example)

```bash
python train_vla.py \
  --batch_size 16 \
  --epochs 10 \
  --lr 1e-3 \
  --checkpoint_dir checkpoints

  python inference_vla.py \
  --checkpoint checkpoints/vla_covla_best.pt \
  --image path/to/image.png \
  --description "The ego vehicle is approaching a crosswalk with a pedestrian nearby."


# ROB-535-Self-Driving-Cars-Perception-and-Controls
Project
