# AMD Gaussian Splatting — ROCm 7.2 / gfx1201

This repository provides a tested HIP/ROCm port of the training extensions used
by the official Graphdeco 3D Gaussian Splatting implementation.

## Scope

The following training components have been ported:

- `diff-gaussian-rasterization`
- `simple-knn`
- `fused-ssim`

The port targets headless training, evaluation and rendering. The SIBR
CUDA/OpenGL viewer has not been ported.

## Validated platform

- AMD Radeon AI PRO R9700
- GPU architecture: `gfx1201`
- ROCm 7.2
- PyTorch 2.13.0+rocm7.2
- Torchvision 0.28.0+rocm7.2
- Python 3.12
- Ubuntu Linux

Other AMD architectures may work, but have not been validated.

## Clone

```bash
git clone --recursive \
  https://github.com/Painter3000/amd-gaussian-splatting-rocm72-gfx1201.git

cd amd-gaussian-splatting-rocm72-gfx1201
```

## Check the ROCm environment

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

hipcc --version
```

## Build the training extensions

Build isolation must be disabled because the extension setup scripts import
PyTorch during the build.

```bash
for sm in \
  submodules/diff-gaussian-rasterization \
  submodules/simple-knn \
  submodules/fused-ssim
do
  PYTORCH_ROCM_ARCH=gfx1201 \
  MAX_JOBS=4 \
  python -m pip install \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    "$sm" || exit 1
done
```

## Import check

```bash
python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn
import fused_ssim

assert torch.cuda.is_available()
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("ROCM_EXTENSIONS: PASS")
PY
```

## Training example

```bash
python train.py \
  -s /path/to/colmap/dataset \
  -m output/rocm-training \
  --data_device cpu \
  --iterations 30000 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  --disable_viewer
```

PyTorch continues to use the `cuda` device name for ROCm devices. Code such as
`torch.device("cuda")` is therefore expected and does not indicate that CUDA is
being used.

## Validation results

The port was validated on the Tanks and Temples `truck` scene.

### Extension validation

* Rasterizer forward and backward kernels: passed
* Multi-case rasterizer stability runs: passed
* `simple-knn` checked against a PyTorch/CPU reference:
  maximum absolute difference `1.71e-07`
* `fused-ssim` checked against the repository reference implementation:
  value error approximately `3e-08`, gradient relative L2 approximately `4e-07`

### End-to-end training

A 30,000-iteration training run completed successfully:

| Metric                        |    Result |
| ------------------------------ | --------: |
| Wall-clock time                |  12:45.50 |
| Final Gaussian count           | 1,141,849 |
| Test PSNR                      |   26.7258 |
| Test SSIM                      |  0.927555 |
| Test LPIPS                     |  0.072381 |
| Peak VRAM                      |  3.68 GiB |
| Maximum edge temperature       |      65 C |
| Maximum junction temperature   |      83 C |
| Maximum memory temperature     |      86 C |

Rendering produced all 32 expected test images successfully.

## Known limitations

* Only `gfx1201` has been validated.
* The SIBR viewer remains on its original CUDA/OpenGL implementation.
* No full CUDA-versus-ROCm tensor parity study has been performed.
* ROCm-SMI power readings on the tested RDNA4 system included values above the
  hardware power cap and are not used for efficiency claims. Related ROCm/
  gfx1201 telemetry issues exist upstream (e.g. ROCm/ROCm#5849, ROCm/ROCm#6289),
  though the exact mechanism behind these specific readings has not been
  independently confirmed.
* The optional Graphdeco `3dgs_accel` rasterizer and Taming-3DGS budget system
  are not included in this baseline port.

## Upstream projects

* [Graphdeco Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn)
* [fused-ssim](https://github.com/rahul-goel/fused-ssim)

## License

This repository is a derived port of the upstream projects. Their original
copyright notices and license terms remain in effect. Consult the license files
in the main repository and each submodule before redistribution or commercial
use.
