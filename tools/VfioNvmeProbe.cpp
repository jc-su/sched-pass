#include "nta/NvmeRuntime.h"

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
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

const char *hbmBackendName(nta::NvmeHbmMappingBackend backend) {
  switch (backend) {
  case nta::NvmeHbmMappingBackend::Unavailable:
    return "unavailable";
  case nta::NvmeHbmMappingBackend::NvidiaPeerPages:
    return "nvidia-peer-pages";
  }
  return "unknown";
}

std::optional<std::string> discoverVfioNvmeEndpoint() {
  const std::filesystem::path devices("/sys/bus/pci/devices");
  std::error_code error;
  if (!std::filesystem::is_directory(devices, error) || error) {
    return std::nullopt;
  }
  for (const auto &entry : std::filesystem::directory_iterator(devices, error)) {
    if (error || !entry.is_directory(error)) {
      continue;
    }
    const std::filesystem::path device = entry.path();
    std::ifstream classFile(device / "class");
    std::string classCode;
    if (!(classFile >> classCode) || !classCode.starts_with("0x0108")) {
      continue;
    }
    std::error_code driverError;
    const std::filesystem::path driver =
        std::filesystem::read_symlink(device / "driver", driverError);
    if (driverError || driver.filename() != "vfio-pci") {
      continue;
    }
    bool hasNamespace = false;
    const std::filesystem::path controllerRoot = device / "nvme";
    std::error_code namespaceError;
    if (std::filesystem::is_directory(controllerRoot, namespaceError) &&
        !namespaceError) {
      for (const auto &controller : std::filesystem::directory_iterator(
               controllerRoot, namespaceError)) {
        if (namespaceError) {
          break;
        }
        std::error_code entryError;
        if (!std::filesystem::is_directory(controller, entryError) ||
            entryError) {
          continue;
        }
        for (const auto &namespaceEntry : std::filesystem::directory_iterator(
                 controller.path(), entryError)) {
          if (entryError) {
            break;
          }
          const std::string name = namespaceEntry.path().filename().string();
          if (name.starts_with("nvme") &&
              name.find('n', std::string("nvme").size()) !=
                  std::string::npos) {
            hasNamespace = true;
            break;
          }
        }
        if (hasNamespace) {
          break;
        }
      }
    }
    if (hasNamespace) {
      return "vfio:" + device.filename().string();
    }
  }
  return std::nullopt;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 7) {
      throw std::invalid_argument(
          "usage: nta-vfio-nvme-probe [vfio:DDDD:BB:SS.F] [gpu] [nsid] [depth] "
          "[hardware-write-protect|trusted-read-only-code] "
          "[hbm-peer|host-mapped]");
    }
    nta::NvmeTransportOptions options;
    const char *configuredEndpoint = std::getenv("NTA_NVME_ENDPOINT");
    if (argc > 1) {
      options.endpoint = argv[1];
    } else if (configuredEndpoint != nullptr && *configuredEndpoint != '\0') {
      options.endpoint = configuredEndpoint;
    } else {
      const char *configuredBdf = std::getenv("NTA_NVME_BDF");
      if (configuredBdf != nullptr && *configuredBdf != '\0') {
        options.endpoint = std::string(configuredBdf).starts_with("vfio:")
                               ? configuredBdf
                               : "vfio:" + std::string(configuredBdf);
      } else {
        options.endpoint = discoverVfioNvmeEndpoint().value_or(std::string{});
      }
    }
    if (options.endpoint.empty()) {
      std::cerr << "nta-vfio-nvme-probe skipped: no VFIO-bound NVMe "
                   "controller with a namespace was discovered; bind a "
                   "dedicated controller or set NTA_NVME_ENDPOINT\n";
      return 77;
    }
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
    if (argc > 6) {
      const std::string_view target = argv[6];
      if (target == "hbm-peer") {
        options.dmaTarget = nta::NvmeDmaTarget::HbmPeer;
      } else if (target == "host-mapped") {
        options.dmaTarget = nta::NvmeDmaTarget::HostMapped;
      } else {
        throw std::invalid_argument("invalid DMA target");
      }
    }
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error("cuInit failed");
    }

    const nta::NvmeDmaTarget dmaTarget = options.dmaTarget;
    nta::NvmeTransport transport(std::move(options));
    const nta::NvmeCapabilities &capabilities = transport.capabilities();
    if (!capabilities.translatedIommu ||
        !capabilities.gpuDoorbellMappingValidated) {
      throw std::runtime_error(
          "VFIO NVMe containment/compatibility gate failed");
    }
    // Exercise the selected data-plane mapping without issuing an application
    // I/O command. HbmPeer covers CUDA HBM -> NVIDIA peer pages -> NVMe PRP;
    // HostMapped is the matched pinned-host baseline. Transport construction
    // has already performed its bounded bootstrap READ into control-plane host
    // memory. End-to-end I/O into the selected destination is deliberately a
    // stronger gate and belongs to nta-nvme-bench / the qualification runner.
    auto probeBuffer = transport.allocate(capabilities.lbaSize);
    if (probeBuffer->dmaTarget() != dmaTarget ||
        probeBuffer->dmaPageCount() == 0) {
      throw std::runtime_error("NVMe DMA mapping qualification failed");
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
              << " destination="
              << (dmaTarget == nta::NvmeDmaTarget::HbmPeer
                      ? "hbm-peer"
                      : "host-mapped")
              << " hbm_peer_dma="
              << (capabilities.supportsHbmPeerDma ? "available"
                                                     : "unavailable")
              << " hbm_mapping_backend="
              << hbmBackendName(capabilities.hbmMappingBackend)
              << " hbm_dma_map="
              << (dmaTarget == nta::NvmeDmaTarget::HbmPeer
                      ? "validated"
                      : "not-applicable")
              << " gpu_control_path=validated"
              << " selected_data_io=not-exercised\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "nta-vfio-nvme-probe failed: " << error.what() << '\n';
    return 1;
  }
}
