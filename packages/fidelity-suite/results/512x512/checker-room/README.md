A floor and two walls with a nearest-filtered 16x16 checker texture, thin bright rods and a small cube under a point light: high-frequency texture and sub-pixel lines.

Valid size 512x512, padded field 576x512, pooled levels 288x256, 144x128, 72x64, 36x32, 20x16, 12x8, 96 ViT tokens. Synthetic weights: the images are parity evidence, not pictures. Device: nvidia ampere NVIDIA GeForce RTX 3060 Ti, Chrome (D3D12 + DXC, hardware f16).

## Exactness against the standalone reference

- **three-dlss-nr: native TSL port**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.
- **three-dlss-nr: reference WGSL shim**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.

The 84 tensors: the input features each renderer ran on, all 79 captured block boundaries, the three post tensors and the f32 head (verdicts and SHA-256 digests in exactness.json).

## GPU time per frame

- **OpenDLSS-NR (reference WebGPU port, standalone)**: median 124.4 ms (min 124.0, wall 125.4), 1.00x the reference
- **three-dlss-nr: native TSL port**: median 201.0 ms (min 200.5, wall 201.8), 1.62x the reference
- **three-dlss-nr: reference WGSL shim**: median 124.2 ms (min 124.0, wall 125.0), 1.00x the reference

Whole network, 451 dispatches, no boundary captures; 10 frames after 3 warm-up frames, renderers interleaved frame by frame, timestamp queries around the frame.
