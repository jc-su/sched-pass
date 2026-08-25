#include "nta/CxlDaxDiscovery.h"

#include <sys/stat.h>

#include <filesystem>
#include <iostream>
#include <string>

int main() {
  const auto endpoint = nta::qualification::discoverDaxEndpoint();
  if (!endpoint.has_value()) {
    // A platform without a CXL Type-3 region is still a valid discovery
    // result.  Physical mapping remains a separate qualification gate.
    std::cout << "cxl_dax_discovery=pass candidate=none\n";
    return 0;
  }

  const std::filesystem::path path(*endpoint);
  struct stat status {};
  if (!path.filename().string().starts_with("dax") ||
      !nta::qualification::isCxlDaxNode(path.filename().string()) ||
      ::stat(path.c_str(), &status) != 0 || !S_ISCHR(status.st_mode)) {
    std::cerr << "CXL DAX discovery returned an invalid endpoint\n";
    return 1;
  }
  std::cout << "cxl_dax_discovery=pass candidate=" << *endpoint << '\n';
  return 0;
}
