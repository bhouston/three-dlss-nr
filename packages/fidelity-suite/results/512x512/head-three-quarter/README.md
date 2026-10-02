The Lee Perry-Smith head scan (three-quarter view) in the demo's studio: physically based skin with colour, normal and specular maps, four directional lights and a dim room environment, rendered by three.js WebGPURenderer into an RGBA16F target.

Valid size 512x512, padded field 576x512, pooled levels 288x256, 144x128, 72x64, 36x32, 20x16, 12x8, 96 ViT tokens. Synthetic weights: the images are parity evidence, not pictures. Device: nvidia ampere NVIDIA GeForce RTX 3060 Ti, Chrome (D3D12 + DXC, hardware f16).

## Exactness against the standalone reference

- **three-dlss-nr: native TSL port**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.
- **three-dlss-nr: reference WGSL shim**: bit-exact. Raw tensors 84/84 byte-identical, PNGs 8/8 byte-identical.

The 84 tensors: the input features each renderer ran on, all 79 captured block boundaries, the three post tensors and the f32 head (verdicts and SHA-256 digests in exactness.json).

## GPU time per frame

- **OpenDLSS-NR (reference WebGPU port, standalone)**: median 123.7 ms (min 123.3, wall 125.0), 1.00x the reference
- **three-dlss-nr: native TSL port**: median 201.0 ms (min 199.5, wall 201.9), 1.62x the reference
- **three-dlss-nr: reference WGSL shim**: median 123.8 ms (min 123.2, wall 125.1), 1.00x the reference

Whole network, 451 dispatches, no boundary captures; 10 frames after 3 warm-up frames, renderers interleaved frame by frame, timestamp queries around the frame.

Head: "Infinite, 3D Head Scan" by Lee Perry-Smith ([Infinite-Realities](https://ir-ltd.net)), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), via the three.js examples.
