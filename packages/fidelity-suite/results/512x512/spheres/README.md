Twelve PBR spheres and boxes (rough to glossy, dielectric and metal) under a hemisphere and a directional light, with an emissive ring at intensity 6 whose HDR highlights exercise the display proxy shoulder.

Valid size 512x512, padded field 576x512, pooled levels 288x256, 144x128, 72x64, 36x32, 20x16, 12x8, 96 ViT tokens. Synthetic weights: the images are parity evidence, not pictures. Device: nvidia ampere NVIDIA GeForce RTX 3060 Ti, Chrome (D3D12 + DXC, hardware f16).

## Exactness against the standalone reference

- **three-dlss-nr: native TSL port**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.
- **three-dlss-nr: reference WGSL shim**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.

The 84 tensors: the input features each renderer ran on, all 79 captured block boundaries, the three post tensors and the f32 head (verdicts and SHA-256 digests in exactness.json).

## GPU time per frame

- **OpenDLSS-NR (reference WebGPU port, standalone)**: median 124.3 ms (min 123.8, wall 125.3), 1.00x the reference
- **three-dlss-nr: native TSL port**: median 201.0 ms (min 200.8, wall 202.9), 1.62x the reference
- **three-dlss-nr: reference WGSL shim**: median 124.1 ms (min 123.7, wall 125.0), 1.00x the reference

Whole network, 451 dispatches, no boundary captures; 10 frames after 3 warm-up frames, renderers interleaved frame by frame, timestamp queries around the frame.
