#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace nta::detail {

inline void checkCudaDevice(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

inline int resolveCudaDevice(int requested) {
  int count = 0;
  int current = 0;
  checkCudaDevice(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  checkCudaDevice(cudaGetDevice(&current), "cudaGetDevice");
  const int selected = requested < 0 ? current : requested;
  if (selected < 0 || selected >= count) {
    throw std::out_of_range("CUDA device ordinal is not present");
  }
  return selected;
}

class CudaDeviceGuard {
public:
  explicit CudaDeviceGuard(int selected) {
    checkCudaDevice(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != selected) {
      checkCudaDevice(cudaSetDevice(selected), "cudaSetDevice");
      changed_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (changed_)
      (void)cudaSetDevice(previous_);
  }

  CudaDeviceGuard(const CudaDeviceGuard &) = delete;
  CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
  int previous_ = 0;
  bool changed_ = false;
};

class NoexceptCudaDeviceGuard {
public:
  explicit NoexceptCudaDeviceGuard(int selected) noexcept {
    if (cudaGetDevice(&previous_) == cudaSuccess && previous_ != selected &&
        cudaSetDevice(selected) == cudaSuccess) {
      changed_ = true;
    }
  }

  ~NoexceptCudaDeviceGuard() {
    if (changed_)
      (void)cudaSetDevice(previous_);
  }

private:
  int previous_ = 0;
  bool changed_ = false;
};

} // namespace nta::detail
