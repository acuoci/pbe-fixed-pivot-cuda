// =============================================================================
// backend.hpp  --  Backend tags, resource ownership, and reusable workspaces
//
// These types make CPU/CUDA memory and lifetime expectations explicit without
// changing the existing low-level RHS launch functions.
// =============================================================================

#pragma once

#include "pbe_cuda/array_view.hpp"
#include "pbe_cuda/cuda_compat.cuh"
#include "pbe_cuda/sectional_grid.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pbe_cuda {

enum class BackendKind {
    Cpu,
    Cuda
};

struct CpuBackend {
    static constexpr BackendKind kind = BackendKind::Cpu;
};

struct CudaBackend {
    static constexpr BackendKind kind = BackendKind::Cuda;
};

#if defined(PBE_ENABLE_CUDA)
static constexpr bool cuda_backend_available = true;
#else
static constexpr bool cuda_backend_available = false;
#endif

class CpuWorkspace {
public:
    CpuWorkspace() = default;
    explicit CpuWorkspace(std::size_t scratch_size)
    {
        ensure_scratch_size(scratch_size);
    }

    void ensure_scratch_size(std::size_t scratch_size)
    {
        if (scratch_.size() < scratch_size)
            scratch_.resize(scratch_size, 0.0);
    }

    [[nodiscard]] std::size_t scratch_size() const noexcept
    {
        return scratch_.size();
    }

    [[nodiscard]] RealView scratch() noexcept
    {
        return RealView(scratch_.data(), scratch_.size());
    }

    [[nodiscard]] ConstRealView scratch() const noexcept
    {
        return ConstRealView(scratch_.data(), scratch_.size());
    }

private:
    std::vector<double> scratch_;
};

#if defined(PBE_ENABLE_CUDA)

inline void throw_on_cuda_error(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
}

class CudaStream {
public:
    CudaStream() = default;

    static CudaStream create(unsigned int flags = cudaStreamNonBlocking)
    {
        CudaStream result;
        throw_on_cuda_error(cudaStreamCreateWithFlags(&result.stream_, flags),
                            "cudaStreamCreateWithFlags");
        result.owns_ = true;
        return result;
    }

    static CudaStream external(cudaStream_t stream) noexcept
    {
        CudaStream result;
        result.stream_ = stream;
        result.owns_ = false;
        return result;
    }

    ~CudaStream()
    {
        if (owns_ && stream_)
            cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(other.stream_), owns_(other.owns_)
    {
        other.stream_ = nullptr;
        other.owns_ = false;
    }

    CudaStream& operator=(CudaStream&& other) noexcept
    {
        if (this != &other) {
            if (owns_ && stream_)
                cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            owns_ = other.owns_;
            other.stream_ = nullptr;
            other.owns_ = false;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] bool owns_stream() const noexcept { return owns_; }

private:
    cudaStream_t stream_ = nullptr;
    bool owns_ = false;
};

template <typename T>
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    explicit CudaDeviceBuffer(std::size_t size)
    {
        resize(size);
    }

    ~CudaDeviceBuffer()
    {
        release();
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void resize(std::size_t size)
    {
        if (size == size_)
            return;

        release();
        if (size == 0)
            return;

        throw_on_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&ptr_), size * sizeof(T)),
            "cudaMalloc");
        size_ = size;
    }

    void upload(ArrayView<const T> host, cudaStream_t stream = 0)
    {
        if (host.size() != size_)
            throw std::invalid_argument(
                "CudaDeviceBuffer: host view size does not match device buffer");
        throw_on_cuda_error(
            cudaMemcpyAsync(ptr_, host.data(), size_ * sizeof(T),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(host-to-device)");
    }

    void download(ArrayView<T> host, cudaStream_t stream = 0) const
    {
        if (host.size() != size_)
            throw std::invalid_argument(
                "CudaDeviceBuffer: host view size does not match device buffer");
        throw_on_cuda_error(
            cudaMemcpyAsync(host.data(), ptr_, size_ * sizeof(T),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(device-to-host)");
    }

    void zero(cudaStream_t stream = 0)
    {
        throw_on_cuda_error(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream),
                            "cudaMemsetAsync");
    }

    void release() noexcept
    {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

class CudaDeviceGrid {
public:
    explicit CudaDeviceGrid(const SectionalGrid& grid, cudaStream_t stream = 0)
        : n_(grid.n()),
          log_x0_(grid.log_x0()),
          inv_log_r_(grid.inv_log_r()),
          pivots_(grid.size())
    {
        pivots_.upload(grid.pivots(), stream);
    }

    [[nodiscard]] const double* data() const noexcept { return pivots_.data(); }
    [[nodiscard]] int n() const noexcept { return n_; }
    [[nodiscard]] double log_x0() const noexcept { return log_x0_; }
    [[nodiscard]] double inv_log_r() const noexcept { return inv_log_r_; }
    [[nodiscard]] std::size_t size() const noexcept { return pivots_.size(); }

private:
    int n_ = 0;
    double log_x0_ = 0.0;
    double inv_log_r_ = 0.0;
    CudaDeviceBuffer<double> pivots_;
};

class CudaWorkspace {
public:
    CudaWorkspace()
        : stream_(CudaStream::create())
    {}

    explicit CudaWorkspace(CudaStream stream)
        : stream_(std::move(stream))
    {}

    void ensure_scratch_size(std::size_t scratch_size)
    {
        if (scratch_.size() < scratch_size)
            scratch_.resize(scratch_size);
    }

    [[nodiscard]] cudaStream_t stream() const noexcept
    {
        return stream_.get();
    }

    [[nodiscard]] CudaDeviceBuffer<double>& scratch() noexcept
    {
        return scratch_;
    }

    [[nodiscard]] const CudaDeviceBuffer<double>& scratch() const noexcept
    {
        return scratch_;
    }

private:
    CudaStream stream_;
    CudaDeviceBuffer<double> scratch_;
};

#endif // defined(PBE_ENABLE_CUDA)

} // namespace pbe_cuda
