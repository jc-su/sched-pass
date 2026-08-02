#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nta::benchmark {

inline void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

inline void checkDriver(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  const char *description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + ": " +
      (name == nullptr ? "unknown CUDA driver error" : name) + " (" +
      (description == nullptr ? "no description" : description) + ")");
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) {
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
        "cudaMalloc benchmark device buffer");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  [[nodiscard]] T *get() const noexcept { return pointer_; }

private:
  T *pointer_ = nullptr;
};

template <typename T> class PinnedBuffer {
public:
  explicit PinnedBuffer(std::size_t count) {
    checkCuda(
        cudaMallocHost(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
        "cudaMallocHost benchmark pinned buffer");
  }
  ~PinnedBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
  }
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  [[nodiscard]] T *get() const noexcept { return pointer_; }

private:
  T *pointer_ = nullptr;
};

} // namespace nta::benchmark
