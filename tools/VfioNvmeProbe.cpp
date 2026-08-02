#include "nta/NvmeRuntime.h"

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

std::uint32_t parse(std::string_view value, const char *name, bool allowZero) {
  char *end = nullptr;
  const std::string storage(value);
  const unsigned long long parsed = std::strtoull(storage.c_str(), &end, 10);
  if (end == storage.c_str() || *end != '\0' || (!allowZero && parsed == 0) ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return static_cast<std::uint32_t>(parsed);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 6) {
      throw std::invalid_argument(
          "usage: nta-vfio-nvme-probe vfio:DDDD:BB:SS.F [gpu] [nsid] [depth] "
          "[hardware-write-protect|trusted-read-only-code]");
    }
    nta::NvmeTransportOptions options;
    options.endpoint = argv[1];
    if (argc > 2) {
      const std::uint32_t gpu = parse(argv[2], "GPU ordinal", true);
      if (gpu > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GPU ordinal exceeds int");
      }
      options.deviceOrdinal = static_cast<int>(gpu);
    }
    if (argc > 3) {
      options.namespaceId = parse(argv[3], "namespace ID", false);
    }
    if (argc > 4) {
      options.queueDepth = parse(argv[4], "queue depth", false);
    }
    if (argc > 5) {
      const std::string_view policy = argv[5];
      if (policy == "hardware-write-protect") {
        options.mediaPolicy =
            nta::NvmeMediaPolicy::RequireHardwareWriteProtection;
      } else if (policy == "trusted-read-only-code") {
        options.mediaPolicy = nta::NvmeMediaPolicy::TrustReadOnlyDeviceCode;
      } else {
        throw std::invalid_argument("invalid media policy");
      }
    }
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("cuInit failed");
    }

    nta::NvmeTransport transport(std::move(options));
    const nta::NvmeCapabilities &capabilities = transport.capabilities();
    if (!capabilities.translatedIommu ||
        !capabilities.gpuDoorbellMappingValidated) {
      throw std::runtime_error(
          "VFIO NVMe containment/compatibility gate failed");
    }
    std::cout << "vfio_nvme_probe=passed"
              << " gpu=" << capabilities.deviceOrdinal
              << " queue_depth=" << capabilities.queueDepth
              << " queue_count=" << capabilities.queueCount
              << " lba_size=" << capabilities.lbaSize
              << " max_transfer_bytes=" << capabilities.maxTransferBytes
              << " namespace_bytes=" << capabilities.namespaceBytes
              << " iommu=translated namespace_policy="
              << (capabilities.namespaceReadOnly ? "hardware-write-protected"
                                                 : "trusted-read-only-code")
              << " gpu_nvme_path=validated\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "nta-vfio-nvme-probe failed: " << error.what() << '\n';
    return 1;
  }
}
