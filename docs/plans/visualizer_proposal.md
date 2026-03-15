# Audio Visualizer Proposal

## Overview

Real-time audio visualizer that maps SyneJEPA audio embeddings to colors. This is a separate deliverable from the core training/eval pipeline.

## Status: TODO

Detailed design to be completed after MVP training pipeline is functional.

## Key Design Questions

1. **Display framework**: pygame, tkinter, web-based (WebSocket + canvas), or LED strip output?
2. **Color mapping**: PCA to 3D → RGB? Learned mapping? UMAP?
3. **Update rate**: 4 Hz (0.25s hop) vs higher for smoother transitions?
4. **Smoothing**: EMA alpha value, interpolation method
5. **Input**: Microphone capture vs. file playback vs. both?
6. **Latency budget**: ViT-S forward ~20ms on CPU, well within 250ms target

## Preliminary Architecture

- `embedder.py` — PyAudio mic capture, 4s sliding window, frozen encoder inference
- `color_map.py` — PCA(3) fitted on calibration set, scale to [0,255], EMA smoothing
- `app.py` — Display layer (framework TBD)
