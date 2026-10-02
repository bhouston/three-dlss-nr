A floor and two walls with a nearest-filtered 16x16 checker texture, thin bright rods and a small cube under a point light: high-frequency texture and sub-pixel lines.

Valid size 256x256, padded field 320x320, pooled levels 160x160, 80x80, 40x40, 20x20, 12x12, 8x8, 64 ViT tokens. Synthetic weights: the images are parity evidence, not pictures. Device: nvidia ampere NVIDIA GeForce RTX 3060 Ti, Chrome (D3D12 + DXC, hardware f16).

## Exactness against the standalone reference

- **three-dlss-nr: native TSL port**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.
- **three-dlss-nr: reference WGSL shim**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.

The 84 tensors: the input features each renderer ran on, all 79 captured block boundaries, the three post tensors and the f32 head (verdicts and SHA-256 digests in exactness.json).

## GPU time per frame

- **OpenDLSS-NR (reference WebGPU port, standalone)**: median 56.9 ms (min 56.7, wall 57.8), 1.00x the reference
- **three-dlss-nr: native TSL port**: median 92.4 ms (min 92.2, wall 93.5), 1.62x the reference
- **three-dlss-nr: reference WGSL shim**: median 57.1 ms (min 56.7, wall 58.3), 1.00x the reference

Whole network, 451 dispatches, no boundary captures; 10 frames after 3 warm-up frames, renderers interleaved frame by frame, timestamp queries around the frame.
