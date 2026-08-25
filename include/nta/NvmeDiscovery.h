#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>

namespace nta::qualification {

struct NvmeDiscoveryRoots {
  std::filesystem::path pciDevicesRoot = "/sys/bus/pci/devices";
};

// Discovery is deliberately read-only.  A candidate is usable by the
// transport only after an explicit VFIO rebind and the transport's own
// translated-IOMMU, namespace-policy, and GPU-doorbell checks.
inline std::optional<std::string> discoverVfioNvmeEndpoint(
    const NvmeDiscoveryRoots &roots = NvmeDiscoveryRoots{}) {
  const std::filesystem::path devices = roots.pciDevicesRoot;
  std::error_code error;
  if (!std::filesystem::is_directory(devices, error) || error) {
    return std::nullopt;
  }
  for (const auto &entry :
       std::filesystem::directory_iterator(devices, error)) {
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
    const std::filesystem::path controllerRoot = device / "nvme";
    std::error_code namespaceError;
    if (!std::filesystem::is_directory(controllerRoot, namespaceError) ||
        namespaceError) {
      continue;
    }
    for (const auto &controller :
         std::filesystem::directory_iterator(controllerRoot, namespaceError)) {
      if (namespaceError) {
        break;
      }
      std::error_code entryError;
      if (!std::filesystem::is_directory(controller, entryError) ||
          entryError) {
        continue;
      }
      for (const auto &namespaceEntry :
           std::filesystem::directory_iterator(controller.path(), entryError)) {
        if (entryError) {
          break;
        }
        const std::string name = namespaceEntry.path().filename().string();
        if (name.starts_with("nvme") &&
            name.find('n', std::string("nvme").size()) != std::string::npos) {
          return "vfio:" + device.filename().string();
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace nta::qualification
