# V-JEPA

Video Joint Embedding Predictive Architecture (V-JEPA) pre-training using Meta's JEPA framework.
Trains a ViT-Huge/16 video encoder to predict masked spatiotemporal representations.

## What it measures

- GPU throughput for video self-supervised learning (ViT-Huge encoder + predictor)
- Video decoding and data loading pressure (16-frame clips at 224x224)
- Multi-GPU scaling via DDP with static graph optimization

## Framework

Wraps [facebookresearch/jepa](https://github.com/facebookresearch/jepa) (pinned commit).
The main training loop is reimplemented inline in `main.py` rather than importing the upstream entrypoint.

## Config variants

| Name | Scale | Notes |
|------|-------|-------|
| `vjepa-single` | per-GPU | batch 24, 12 workers |
| `vjepa-gpus` | all GPUs, 1 node | batch 24 per GPU |

## Data

Synthetic random MP4 videos (640x480, 300 frames, 30fps) generated at prepare time.
1000 videos by default, listed in a CSV manifest (`video_metainfo.csv`).

## Key dependencies

- JEPA source (cloned into `jepa/`)
- PyTorch DDP (DistributedDataParallel, static_graph=True)
- torchcompat (device abstraction, process group)
- OpenCV (cv2) for synthetic video generation
- benchmate (observer, monitoring)
