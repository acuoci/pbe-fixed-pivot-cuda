# Build Guide

## Requirements

| Component | Minimum version | Notes |
|---|---|---|
| CUDA Toolkit | 12.0 | nvcc must be in PATH or loaded via module |
| CMake | 3.18 | First version with stable native CUDA language support |
| C++ standard | C++17 | Host code; device code compiled as C++17 |
| GPU compute capability | sm_80 or higher | A100, A30 (sm_80); RTX 30xx, A40 (sm_86); H100 (sm_90) |
| Host compiler | GCC 9+ | Must be compatible with the installed CUDA Toolkit |

> **Note:** The library uses C++17 consistently for host and device-facing code
> to maximize compatibility with CUDA/HPC compiler combinations.

---

## Quick build

```bash
git clone git@github.com:YOUR_USERNAME/pbe-fixed-pivot-cuda.git
cd pbe-fixed-pivot-cuda
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86"

cmake --build . -j$(nproc)
```

---

## CMake options

| Option | Default | Description |
|---|---|---|
| `PBE_BUILD_STATIC` | `ON` | Build static library (`libpbe_cuda.a`) |
| `PBE_BUILD_SHARED` | `OFF` | Build shared library (`libpbe_cuda.so`) |
| `PBE_BUILD_EXAMPLES` | `ON` | Build all worked examples |
| `PBE_BUILD_TESTS` | `ON` | Build GoogleTest regression suite |
| `PBE_ENABLE_WARNINGS` | `ON` | Enable `-Wall -Wextra` for host and device code |
| `CMAKE_CUDA_ARCHITECTURES` | `80 86` | Target GPU architectures |
| `GTEST_ROOT` | _(unset)_ | Path to GoogleTest installation (if not in standard paths) |

---

## GPU architecture selection

**This is the most important build option.** The correct architecture must match your GPU hardware:

| GPU | Compute capability | CMake value |
|---|---|---|
| NVIDIA A30 | sm_80 | `80` |
| NVIDIA A100 | sm_80 | `80` |
| NVIDIA H100 | sm_90 | `90` |
| RTX 3090 / A40 | sm_86 | `86` |
| RTX 4090 | sm_89 | `89` |

Setting the wrong architecture will cause a `named symbol not found` runtime error. To target multiple architectures (e.g. for a shared cluster):

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
```

> **Critical:** `CMAKE_CUDA_ARCHITECTURES` must be set **before** `project()` in `CMakeLists.txt` to override CMake's default. The provided `CMakeLists.txt` handles this correctly; if you override from the command line you must also wipe the build directory first (`rm -rf build`).

---

## HPC cluster build (modules)

On a typical HPC cluster with environment modules:

```bash
# Load required modules (adjust to your cluster's module names)
module load cuda/12.6
module load gcc/9.2.0
module load cmake/4.0

# Verify
nvcc --version
gcc  --version
cmake --version

# Configure and build
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80"   # match your cluster's GPU

cmake --build . -j$(nproc)
```

---

## Build types

```bash
# Release (recommended for benchmarks and production)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Debug (adds -g, disables optimisation — slow but useful for debugging)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release with debug info (optimised + symbols — useful for profiling)
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

---

## Building only the library (no examples or tests)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPBE_BUILD_EXAMPLES=OFF \
    -DPBE_BUILD_TESTS=OFF

cmake --build . --target pbe_cuda_static -j$(nproc)
```

---

## Running the regression tests

```bash
# Build with tests enabled (default)
cmake .. -DPBE_BUILD_TESTS=ON

# If GoogleTest is not in standard paths (e.g. HPC cluster):
cmake .. -DPBE_BUILD_TESTS=ON -DGTEST_ROOT=/path/to/googletest

cmake --build . -j$(nproc)

# Run all tests
ctest --output-on-failure -V

# Run a specific test suite
./tests/test_aggregation
./tests/test_breakage
```

Expected output: 31 tests passing across aggregation and breakage suites.

---

## Integration into your own CMake project

### Option A — FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(pbe_cuda
    GIT_REPOSITORY https://github.com/YOUR_USERNAME/pbe-fixed-pivot-cuda.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(pbe_cuda)

target_link_libraries(your_target PRIVATE pbe_cuda::pbe_cuda)
```

### Option B — find_package (after install)

```bash
# Install to a prefix
cmake --install build --prefix /path/to/install
```

```cmake
find_package(pbe_cuda REQUIRED)
target_link_libraries(your_target PRIVATE pbe_cuda::pbe_cuda)
```

### Option C — Direct inclusion (no CMake)

```bash
# Compile your code
nvcc -std=c++17 -arch=sm_80 \
     -I/path/to/pbe-fixed-pivot-cuda/include \
     your_code.cu \
     /path/to/build/libpbe_cuda.a \
     -lcudart -o your_binary
```

---

## VS Code + SSH setup (recommended for HPC development)

1. Install the **Remote - SSH** extension in VS Code
2. Open Command Palette (`Ctrl+Shift+P`) → **Remote-SSH: Add New SSH Host**
3. Enter: `ssh your_username@your.cluster.address`
4. Connect → **File → Open Folder** → navigate to the repo

Install these extensions **on the remote**:
- **CMake Tools** (`ms-vscode.cmake-tools`) — configure and build from VS Code
- **clangd** (`llvm-vs-code-extensions.vscode-clangd`) — IntelliSense for CUDA/C++
- **GitLens** (`eamodio.gitlens`) — enhanced git history

CMake Tools uses `build/compile_commands.json` (generated automatically by `CMAKE_EXPORT_COMPILE_COMMANDS=ON`) to provide IntelliSense for all source files.

---

## Troubleshooting

**`named symbol not found` at runtime:**
The kernel was compiled for a different architecture than the GPU you are running on. Check `CMAKE_CUDA_ARCHITECTURES` matches your hardware. Wipe the build directory and reconfigure.

**`atomicAdd(double*, double)` compilation error:**
The CUDA architecture flag is not being passed to nvcc. Ensure `CMAKE_CUDA_ARCHITECTURES` is set before `project()` in `CMakeLists.txt`. See the note in the GPU architecture section above.

**`nvcc warning: -std=c++20 flag not supported`:**
The build should not request C++20. Reconfigure from a clean build directory and
verify `CMAKE_CXX_STANDARD=17` and `CMAKE_CUDA_STANDARD=17`.

**GoogleTest not found:**
Either set `-DGTEST_ROOT=/path/to/gtest` to point to an existing installation, or ensure internet access is available for FetchContent to download it automatically.

**Build fails with `target_compile_features no known features for CXX compiler`:**
The system GoogleTest version is too new for the host GCC. Pin to GoogleTest `release-1.12.1` or earlier, or set `-DGTEST_ROOT` to a compatible installation.
