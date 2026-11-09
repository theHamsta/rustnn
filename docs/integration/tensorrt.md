# TensorRT-RTX Backend

The `trtx` backend runs WebNN graphs on NVIDIA RTX GPUs through
[TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) using the
[trtx](https://github.com/rustnn/trtx-rs) crate. Graphs are lowered directly into a TensorRT
network by `src/converters/trtx.rs` (with `trtx_gru.rs`, `trtx_lstm.rs` and `trtx_rnn.rs` for the
recurrent operations); there is no ONNX intermediate. The backend itself is
`src/backends/trtx.rs`.

## Features

| Feature | Use |
|---|---|
| `trtx-runtime` | The backend and converter; pulls in `trtx`, `cudarc` and `zstd-cache-compression` |
| `trtx-converter` | Builds serialized engines through the NVIDIA SDK without rustnn dispatch |
| `trtx-enterprise` | `trtx-runtime` linked against full TensorRT 10 (`nvinfer`, `nvonnxparser`) instead of the TensorRT-RTX libraries; RTX-only features such as CUDA graphs and the JIT runtime cache are compiled out. For validation only |
| `zstd-cache-compression` | Compresses the on-disk caches; enabled by `trtx-runtime` |

## Requirements

- An NVIDIA RTX GPU with a current driver. The CUDA driver API is loaded at run time through
  `cudarc`; the CUDA toolkit is not required. Device enumeration checks for the driver library
  before initializing CUDA. If the library is missing or initialization fails, it reports no
  TensorRT-RTX devices so automatic selection can try another backend.
- The TensorRT-RTX 1.7.1 library. The `trtx` crate loads it by name, so the
  SDK's `lib` directory (Linux) or `bin` directory (Windows) must be on `LD_LIBRARY_PATH` or
  `PATH`, or the application calls `rustnn::backends::trtx::dynamically_load_tensorrt(Some(path))`
  before creating a context. When the library cannot be loaded, backend selection skips
  TensorRT (on Windows the underlying error is `LoadLibraryExW` code 126).
- Build time: `trtx-sys` generates bindings from the TensorRT headers with autocxx and needs
  libclang (`LIBCLANG_PATH` when it is not found automatically). `TENSORRT_INCLUDE_DIR`,
  `TENSORRT_LIB_DIR` and `TENSORRT_SDK_DIR` override the header and library locations.

The Windows steps are in [Windows TensorRT-RTX Setup](https://rustnn.github.io/rustnn/integration/windows-tensorrt-setup/).

## Selection

With `accelerated = true` and the `Default` or `HighPerformance` power preference, TensorRT-RTX
is the first choice when the feature is enabled and a CUDA device is present. Force it with
`MLContextOptions::with_rustnn_backend_hint(Backend::Trtx)` or pick a GPU with
`with_rustnn_device_hint(BackendDevice::Trtx { cuda_device_idx })`. `RUST_LOG=info` prints the
devices found and `Backend selected: Trtx { cuda_device_idx: 0 }`.

## How a graph runs

1. `TrtxConverter::build_network` translates the `GraphInfo` into a TensorRT network. Constants
   become weights and are marked refittable, except constants that TensorRT bakes into the
   engine (for example scalars cast to another type) and constants without a consumer.
2. The engine is built with `kREFIT_INDIVIDUAL` and `kSTRIP_PLAN`, and with TF32 disabled so
   that float32 graphs compute in full float32 precision (the WPT references are strict IEEE
   float32). Float16 operations can promote intermediate values to float32 for numerical
   stability; `reduceLogSum` keeps its sum and logarithm in float32 before casting the result
   back to float16.
3. The stripped engine is stored in the engine cache and then refitted with the actual weights.
   Loading a cached engine skips the build entirely. Engines are cached only when every constant
   is refittable, so weight data never ends up in the cache.
4. Tensors are CUDA device buffers. `dispatch` binds them by name, captures a CUDA graph per
   distinct binding set and replays it on later calls. `write_tensor` synchronizes the stream
   before it returns.

## Native wgpu interoperability

Enable `wgpu-native` together with `trtx-runtime` to share tensor buffers with Vulkan on Linux/Windows
or D3D12 on Windows. Create the context with `GpuDevice::WgpuDevice(WgpuDevice::new(device, queue))`.
The constructor is unsafe: the queue must belong to the device, and all submissions must be
serialized with context creation, export, return and destruction. The CUDA device is matched
by Vulkan device UUID or D3D12 adapter LUID, never by enumeration index.

For Vulkan on Linux or Windows, use
[`create_vulkan_device`](https://rustnn.github.io/rustnn/api/rustnn/backends/trtx/wgpu_interop/fn.create_vulkan_device.html)
to enable the external memory and semaphore extensions: `_fd` on Linux, `_win32` on Windows.
The helper requests `VULKAN_EXTERNAL_MEMORY_FD` or `VULKAN_EXTERNAL_MEMORY_WIN32` respectively.
The device must support timeline semaphores and exportable storage-buffer memory and timeline
semaphores using opaque FDs or NT handles.
Memory flags alone do not enable external semaphore sharing. Existing devices must already
have these capabilities enabled; device creation cannot add extensions retroactively.
Windows uses shared D3D12 committed resources and shared fences on a single-node adapter.
Metal, browser WebGPU and DMA-BUF imports are not implemented by this path.

Use the ownership sequence below. The integration test in `tests/wgpu_trtx_interop.rs` is a
complete example with compute shaders, TensorRT dispatch and graphics readback:

```text
create_exportable_tensor -> ML owns the tensor
export_to_gpu            -> CUDA signals completion; graphics waits/acquires
as_wgpu_buffer (unsafe)  -> submit all buffer uses on the associated wgpu queue
destroy_gpu_buffer       -> graphics releases/signals; CUDA waits; ML owns the tensor
```

Shared buffers support storage and copy usages, with no wgpu mapping. Allocations have fixed
capacity; resizing within that capacity keeps the pointer stable, while growth and capacity
replacement fail. No host staging copy is used between graphics and inference. Allocation and
resource destruction can block; ordinary ownership handoffs enqueue GPU synchronization.
Semaphore waits and signals remain outside TensorRT's captured CUDA graph.

The unsafe buffer-access contract forbids using retained buffers, bind groups or command
buffers after returning ownership, and forbids explicitly destroying the shared wgpu buffer.
Submit every graphics use before returning the export. Dropping a Rust buffer view alone is
not an ownership return. A failed handoff leaves the tensor unavailable to ML; discard the
context. A retained Vulkan buffer holds the native allocation alive through its HAL callback;
D3D12 buffers retain a COM reference to their committed resource.

Run the hardware tests explicitly (they fail if the required hardware/runtime is unavailable):

```bash
export LD_LIBRARY_PATH=/path/to/TensorRT-RTX-1.7.1.116/lib
cargo test --features trtx-runtime,wgpu-native --test wgpu_trtx_interop -- --ignored --nocapture
cargo test --features trtx-runtime,wgpu-native --lib wgpu_interop::tests::vulkan_cuda_shared_buffer -- --ignored --nocapture
```

The first command runs the Vulkan test on Linux and both Vulkan and D3D12 tests on Windows; set `PATH` to the
SDK library directory on Windows. It checks repeated graphics preprocessing, TensorRT ReLU,
graphics postprocessing, input reuse, export ownership rejection and CUDA readback, with CUDA
graphs enabled and disabled. The second command tests Vulkan/CUDA allocation and synchronization
without loading TensorRT. These hardware tests are ignored in ordinary CPU-only test runs.

## Options

Set through `RustNNOptions::trtx` (`TrtxOptions`):

The [TrtxOptions field documentation](https://rustnn.github.io/rustnn/api/rustnn/mlcontextoptions/struct.TrtxOptions.html)
owns defaults and option semantics. For example:

```rust
use rustnn::mlcontext::{Backend, MLContextOptions, MLPowerPreference, RustNNOptions};

let mut options = RustNNOptions::default();
options.trtx.cuda_graphs = false;
let context_options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
    .with_rustnn_backend_hint(Backend::Trtx)
    .with_rustnn_options(options);
```

## Caches

| Category | Location | Content |
|---|---|---|
| `trtx` | `<cache_dir>/rustnn/trtx` | Stripped engines, keyed by the graph topology and a hash of the converter sources |
| `trtx-jit` | `<cache_dir>/rustnn/trtx-jit` | The global TensorRT runtime cache, serialized after builds |

`<cache_dir>` is `~/.cache` on Linux, `~/Library/Caches` on macOS and `%LOCALAPPDATA%` on
Windows. Files are zstd compressed and written through a temporary file plus rename. Deleting
either directory is safe; the next build recreates the entries. A stale engine cannot survive a
converter change because the converter source hash is part of the key.

## Environment variables

| Variable | Effect |
|---|---|
| `RUSTNN_TRTX_LOG_VERBOSITY` | TensorRT logger level: `internal_error`, `error`, `warning`, `info`, `verbose` |
| `TRTX_JSON_DUMP_PATH` | Directory that receives a layer JSON dump per built engine |
| `RUST_LOG` | rustnn logging; `debug` shows cache keys and refit details |

## Testing

```bash
make test-wpt-trtx                                        # WPT suite, requires a GPU
WPT_BACKEND=trtx make test-wpt-op OP=conv2d               # one operation
cargo test --test test_trtx_execution --features trtx-runtime
make wpt-sync-trtx                                        # regenerate PASS snapshots after converter changes
cargo check -F trtx-runtime --all-targets                 # what CI runs
```

There is no GPU runner in CI. The TensorRT PASS snapshots under `tests/snapshots/` are
regenerated by contributors with a GPU and reviewed in the pull request. Debug builds compile an
engine per WPT trial and take tens of minutes for the full suite; use `make test-wpt-op` while
iterating.

## Limitations

- The `shape` extension operation is not lowered.
- Data type restrictions of TensorRT (for example integer element-wise operations) are bridged
  with casts in the converter; cases that still fail show up as missing PASS snapshots and on the
  WPT dashboard.
- Engines are specific to the GPU architecture; the cache is per machine.
