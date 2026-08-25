#include "nta/CxlDaxDiscovery.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main() {
  char temporaryDirectory[] = "/tmp/nta-cxl-discovery-XXXXXX";
  char *createdDirectory = ::mkdtemp(temporaryDirectory);
  if (createdDirectory == nullptr) {
    std::cerr << "cannot create CXL discovery fixture directory\n";
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

  nta::qualification::CxlDaxDiscoveryRoots fixtureRoots;
  fixtureRoots.devRoot = fixtureRoot / "dev";
  fixtureRoots.classRoot = fixtureRoot / "class";
  fixtureRoots.cxlDevicesRoot = fixtureRoot / "cxl";
  std::error_code error;
  std::filesystem::create_directories(
      fixtureRoots.cxlDevicesRoot / "region0", error);
  std::filesystem::create_directories(fixtureRoots.classRoot, error);
  std::filesystem::create_directories(fixtureRoots.devRoot, error);
  std::ofstream(fixtureRoots.cxlDevicesRoot / "region0" / "dax0.0") << "";
  if (error) {
    std::cerr << "cannot create CXL discovery fixture\n";
    return 1;
  }
  std::filesystem::create_symlink(
      fixtureRoots.cxlDevicesRoot / "region0" / "dax0.0",
      fixtureRoots.classRoot / "dax0.0", error);
  std::filesystem::create_symlink("/dev/null", fixtureRoots.devRoot / "dax0.0",
                                  error);
  if (error) {
    std::cerr << "cannot create CXL discovery fixture links\n";
    return 1;
  }
  const auto fixtureEndpoint =
      nta::qualification::discoverDaxEndpoint(fixtureRoots);
  if (!fixtureEndpoint.has_value() ||
      *fixtureEndpoint != (fixtureRoots.devRoot / "dax0.0").string()) {
    std::cerr << "CXL discovery fixture did not find its devdax endpoint\n";
    return 1;
  }

  const auto endpoint = nta::qualification::discoverDaxEndpoint();
  if (!endpoint.has_value()) {
    // The physical machine may not expose a CXL Type-3 region.  The fixture
    // above still proves the real discovery and ancestry checks on every
    // machine; physical mapping remains a separate qualification gate.
    std::cout << "cxl_dax_discovery=pass fixture=devdax candidate=none\n";
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
  std::cout << "cxl_dax_discovery=pass fixture=devdax candidate=" << *endpoint
            << '\n';
  return 0;
}
