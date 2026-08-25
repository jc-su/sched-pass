#include "nta/NvmeDiscovery.h"

#include <iostream>
#include <string>

int main() {
  const auto endpoint = nta::qualification::discoverVfioNvmeEndpoint();
  if (!endpoint.has_value()) {
    // No VFIO candidate is a valid capability result, not a test failure. The
    // physical probe remains separately gated because it would require an
    // explicit controller rebind and a real GPU/NVMe data transfer.
    std::cout << "nvme_discovery=pass candidate=none\n";
    return 0;
  }
  if (!endpoint->starts_with("vfio:") || endpoint->size() <= 5) {
    std::cerr << "NVMe discovery returned an invalid typed endpoint\n";
    return 1;
  }
  std::cout << "nvme_discovery=pass candidate=" << *endpoint << '\n';
  return 0;
}
