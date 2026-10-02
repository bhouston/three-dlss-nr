Three renderers run each scene on the same input features and the same weights: **reference**, the
standalone WebGPU port of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) by
[maan](https://github.com/maanHimself) (MIT, pinned at 9d08f41); **TSL**, the native three.js port
([three-dlss-nr](https://github.com/bhouston/three-dlss-nr)); and **shim**, the reference's own WGSL running inside
three.js on the renderer's device.

**The weights are synthetic** (NVIDIA's weights are not included anywhere), so the outputs are parity evidence, not
pictures. **Bit-exactness is checked on the raw tensors**: each renderer's input features, all 79 block boundaries, the
post tensors and the f32 head are compared byte for byte with the reference (exactness.json in each scene folder). The
images are deterministic views of those tensors, so every pair shows an infinite PSNR, and CI fails if any verdict or
image hash regresses. How the scenes, features and visualizations are made:
[packages/fidelity-suite](https://github.com/bhouston/three-dlss-nr/tree/main/packages/fidelity-suite).

## Summary

GPU time per frame, median, in ms; nvidia ampere NVIDIA GeForce RTX 3060 Ti, Chrome stable (D3D12 + DXC). The shim runs the reference's
WGSL, so it matches the reference's speed; the TSL port computes the same bytes with generated kernels and is about
1.6x slower.

- **Lee Perry-Smith head, front, 256x256**: all bit-exact; GPU ms per frame: reference 56.9, TSL 92.2, shim 56.8
- **Lee Perry-Smith head, three-quarter, 256x256**: all bit-exact; GPU ms per frame: reference 56.8, TSL 92.0, shim 56.9
- **Spheres and an emissive ring, 256x256**: all bit-exact; GPU ms per frame: reference 56.8, TSL 92.4, shim 56.9
- **Checker room with thin lines, 256x256**: all bit-exact; GPU ms per frame: reference 56.9, TSL 92.4, shim 57.1
- **Lee Perry-Smith head, front, 512x512**: all bit-exact; GPU ms per frame: reference 123.3, TSL 199.3, shim 123.3
- **Lee Perry-Smith head, three-quarter, 512x512**: all bit-exact; GPU ms per frame: reference 123.7, TSL 201.0, shim 123.8
- **Spheres and an emissive ring, 512x512**: all bit-exact; GPU ms per frame: reference 124.3, TSL 201.0, shim 124.1
- **Checker room with thin lines, 512x512**: all bit-exact; GPU ms per frame: reference 124.4, TSL 201.0, shim 124.2
