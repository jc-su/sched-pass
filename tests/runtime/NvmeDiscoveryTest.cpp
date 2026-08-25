#include "nta/NvmeDiscovery.h"

#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

int main() {
  char temporaryDirectory[] = "/tmp/nta-nvme-discovery-XXXXXX";
  char *createdDirectory = ::mkdtemp(temporaryDirectory);
  if (createdDirectory == nullptr) {
    std::cerr << "cannot create NVMe discovery fixture directory\n";
    return 1;
  }
  const std::filesystem::path fixtureRoot(createdDirectory);
  struct FixtureCleanup {
    std::filesystem::path root;
    ~FixtureCleanup() {
      std::error_code error;
      std::filesystem::remove_all(root, error);
    }
  } cleanup{fixtureRoot};

  const std::filesystem::path device = fixtureRoot / "0000:d8:00.0";
  std::error_code error;
  std::filesystem::create_directories(device / "nvme" / "nvme7", error);
  if (error) {
    std::cerr << "cannot create NVMe discovery fixture\n";
    return 1;
  }
  std::ofstream(device / "class") << "0x010802\n";
  std::ofstream(device / "nvme" / "nvme7" / "nvme7n1") << "";
  std::filesystem::create_symlink("/sys/bus/pci/drivers/vfio-pci",
                                  device / "driver", error);
  if (error) {
    std::cerr << "cannot create NVMe discovery driver fixture\n";
    return 1;
  }
  const nta::qualification::NvmeDiscoveryRoots fixtureRoots{fixtureRoot};
  const auto fixtureEndpoint =
      nta::qualification::discoverVfioNvmeEndpoint(fixtureRoots);
  if (!fixtureEndpoint.has_value() || *fixtureEndpoint != "vfio:0000:d8:00.0") {
    std::cerr << "NVMe discovery fixture did not find its VFIO namespace\n";
    return 1;
  }

  const auto endpoint = nta::qualification::discoverVfioNvmeEndpoint();
  if (!endpoint.has_value()) {
    // No live VFIO candidate is a valid capability result, not a test failure.
    // The fixture above still executes the full read-only discovery logic on
    // every machine; the physical probe remains separately gated.
    std::cout << "nvme_discovery=pass fixture=vfio candidate=none\n";
    return 0;
  }
  if (!endpoint->starts_with("vfio:") || endpoint->size() <= 5) {
    std::cerr << "NVMe discovery returned an invalid typed endpoint\n";
    return 1;
  }
  std::cout << "nvme_discovery=pass fixture=vfio candidate=" << *endpoint
            << '\n';
  return 0;
}
