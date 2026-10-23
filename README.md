# pixtreme-infer-span

## Purpose

`pixtreme-infer-span` provides the SPAN architecture module for `pixtreme.infer`.
It runs image upscaling from safetensors checkpoints with 48 feature channels
and derives the scale from the checkpoint.

## Supported environment

- Linux x86_64 with glibc 2.28 or newer
- Python 3.12 or newer
- NVIDIA GPU: sm86, sm89, or sm120
- CUDA 13 with a compatible NVIDIA driver, TensorRT 11, and CuPy for CUDA 13

Release wheels contain the native SPAN plugin and cubins for all three GPU
generations. Installation needs no C++ compiler or CUDA/TensorRT SDK setup.

## Installation

Install from PyPI:

```sh
python -m pip install pixtreme-infer-span
```

Wheels and `SHA256SUMS` are also available from the
[GitHub Releases](https://github.com/sync-dev-org/pixtreme-infer-span/releases).
Download the wheel and checksum file for the same version, verify the checksum,
and install the wheel:

```sh
sha256sum -c SHA256SUMS
python -m pip install /path/to/pixtreme_infer_span-X.Y.Z-py3-none-manylinux_2_28_x86_64.whl
```

## Public API

The public module is `pixtreme.infer.span` and exports:

- `FAMILY_CONTRACT`
- `FACTORY`
- `build_engine`
- `SpanArtifactIdentity`

`SpanArtifactIdentity(checkpoint, weight_prefix)` identifies a safetensors
checkpoint. It is an immutable dataclass that converts path-like inputs to
`Path` without resolving relative paths. Equality compares the checkpoint path
and the literal `weight_prefix`.
`weight_prefix=""` selects unprefixed weights; `"params."` selects names under
that prefix. A missing prefix is an error; SPAN never selects another state.

`FAMILY_CONTRACT` declares RGB HWC float32 input and output Frames and no call
parameters. `FACTORY.family_contract` returns that same declaration.
`FACTORY.query_contract(identity, precision=..., input_hw=...)` checks settings
and safetensors header metadata without reading tensor values, importing runtime
backends or allocating GPU memory. The parent pixtreme namespace imports CuPy
during initialization. The query returns a core `StatelessOpenContract` containing
the identity, precision, exact `input_hw=(height, width)` and the checkpoint's
scale. Its `module_contract` property derives the Frame and parameter contract
for that configuration. Precision must be `"fp32"` or `"fp16"`.

SPAN accepts checkpoints with 48 feature channels whose first upsampler
convolution (`upsampler.0.weight`) has shape `(3 * s**2, 48, 3, 3)` for a
positive integer scale `s`. The checkpoint determines `s`; callers do not
specify it. Both precisions produce output height and width equal to the input
height and width multiplied by `s`. Incompatible checkpoints are rejected
before engine creation with their observed and expected shapes.

Prepare PyTorch checkpoints in a separate conversion step:

```sh
pip install 'tensoracle[convert]>=1.1.0'
python -m tensoracle.convert model.pth model.safetensors
```

The converter preserves tensor values and joins nested keys with dots, so a
`params` state becomes the prefix `params.`. SPAN rejects `.pth` inputs with
this conversion instruction. Engine building, saved-engine reuse and inference
use NumPy and safetensors and do not import torch. The inference environment
needs `tensoracle>=1.1.0`, `tensorrt-cu13>=11,<12`, and CuPy (for example
`cupy-cuda13x`, matching the application's CUDA environment).

`FACTORY.open(identity, precision=..., input_hw=...)` repeats the query's checks,
then opens or builds tensoracle's saved engine and returns a reusable Frame
runner. Its `contract` equals the query result for the same settings.
`build_engine(identity, precision=..., input_hw=..., force_rebuild=False)`
prepares the saved engine, closes its execution and returns the checked contract.
Saved engines live under `$XDG_CACHE_HOME/tensoracle`, or
`~/.cache/tensoracle` when `XDG_CACHE_HOME` is unset. Damaged engines or engines
with incompatible saved conditions raise an error; pass `force_rebuild=True` to
`build_engine` to explicitly replace them, then open through `FACTORY.open`.

```python
import cupy as cp
from pixtreme.infer.span import FACTORY, SpanArtifactIdentity, build_engine

identity = SpanArtifactIdentity("/path/to/span.safetensors", "params.")
contract = FACTORY.query_contract(identity, input_hw=(540, 960), precision="fp16")
build_engine(identity, input_hw=(540, 960), precision="fp16")
with FACTORY.open(identity, input_hw=(540, 960), precision="fp16") as runner:
    assert runner.contract == contract
    output_frame = runner(input_frame)
    cp.cuda.get_current_stream().synchronize()
```

Reuse the runner on one CUDA device and stream across frames. `close()` and
leaving a `with` block release its execution context; later calls fail.
Closing does not synchronize CUDA work, so complete pending work before closing.
Output storage is allocated for each call. The public input and output are
RGB HWC float32 Frames; fp16 changes only the network's internal precision.
Tensoracle receives the packaged fp16 plugin through `plugins=` on both build
and reopen. FP32 uses standard TensorRT layers.

For CUDA Graph inference, select `FACTORY.graph_factory()`. The returned
stateless factory accepts the same artifact, precision and exact input size.
Selection and `query_contract` use no GPU resources. Open the instance on its
device inside an explicit non-default CuPy stream:

```python
graph_factory = FACTORY.graph_factory()
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    with graph_factory.open(identity, input_hw=(540, 960), precision="fp16") as runner:
        output_frame = runner(input_frame)  # input_frame must already be ready on stream
        stream.synchronize()
```

Opening warms up a dedicated TensorRT execution and captures one `enqueue`
using fixed input and output storage through core's `CudaGraphSession` and
`CudaGraphBuffers` (`pixtreme-infer>=0.14.1`, `tensoracle>=1.1.0`). Every call
validates its Frame and parameters, copies current pixels into fixed storage,
replays the graph, and copies the output into a separate allocation carrying
that request's metadata. Keep inputs alive until their copies finish; returned
Frames retain their pixels after later calls and after close. Use sequential
calls on the opening device and stream. A different size, device or stream
requires a separate instance. Preparation failure releases its resources and
fails `open`. The factory neither recaptures automatically nor falls back to
ordinary inference. Complete pending work before closing the instance.

## License

The project's own code is licensed under the [MIT License](LICENSE).
The wheel includes third-party license texts and notices in its
`dist-info/licenses/` directory.
