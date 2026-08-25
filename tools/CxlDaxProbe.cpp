#include "nta/CxlRuntime.h"
#include "nta/CxlDaxDiscovery.h"

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int parseInt(std::string_view value, const char *name, int minimum) {
  char *end = nullptr;
  const std::string storage(value);
  const long parsed = std::strtol(storage.c_str(), &end, 10);
  if (end == storage.c_str() || *end != '\0' || parsed < minimum ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return static_cast<int>(parsed);
}

std::string environment(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

} // namespace

int main(int argc, char **argv) {
  bool jsonOutput = false;
  try {
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (argument == "--json=1" || argument == "--json") {
        jsonOutput = true;
      } else if (argument == "--json=0") {
        jsonOutput = false;
      } else {
        positional.push_back(argument);
      }
    }
    if (positional.size() > 3) {
      throw std::invalid_argument(
          "usage: nta-cxl-dax-probe [endpoint] [window-mib] [gpu] [--json=1]");
    }
    const std::string configuredEndpoint =
        !positional.empty() ? positional[0] : environment("NTA_CXL_DAX_DEVICE");
    const std::string endpoint = configuredEndpoint.empty()
                                     ? nta::qualification::discoverDaxEndpoint()
                                           .value_or(std::string{})
                                     : configuredEndpoint;
    if (endpoint.empty()) {
      if (jsonOutput) {
        std::cout << "{\"schema\":1,\"classification\":\"nta-dax-qualification\","
                     "\"tier\":\"dax\",\"status\":\"skipped\","
                     "\"qualified\":false,\"reason\":\"missing_endpoint\"}\n";
      } else {
        std::cerr << "nta-cxl-dax-probe skipped: no /dev/dax* character "
                     "device is exposed; set NTA_CXL_DAX_DEVICE for an "
                     "explicit endpoint\n";
      }
      return 77;
    }
    const int windowMiB =
        parseInt(positional.size() > 1 ? positional[1]
                 : environment("NTA_CXL_DAX_WINDOW_MIB").empty()
                     ? "1024"
                     : environment("NTA_CXL_DAX_WINDOW_MIB"),
                 "window MiB", 1);
    const int deviceOrdinal = parseInt(positional.size() > 2 ? positional[2]
                                       : environment("NTA_CXL_DAX_GPU").empty()
                                           ? "0"
                                           : environment("NTA_CXL_DAX_GPU"),
                                       "GPU ordinal", 0);
    const std::size_t windowBytes =
        static_cast<std::size_t>(windowMiB) * 1024U * 1024U;

    if (cuInit(0) != CUDA_SUCCESS) {
      throw std::runtime_error("cuInit failed");
    }
    nta::CxlDaxOptions options;
    options.endpoint = endpoint;
    options.windowBytes = windowBytes;
    options.deviceOrdinal = deviceOrdinal;
    nta::CxlDaxTransport transport(std::move(options));
    const nta::CxlDaxCapabilities capabilities = transport.capabilities();
    if (!capabilities.hostRegistered || !capabilities.directDeviceVisible) {
      throw std::runtime_error("CXL DAX mapping is not directly CUDA-visible");
    }
    if (jsonOutput) {
      std::cout << "{\"schema\":1,\"classification\":\"nta-dax-qualification\","
                   "\"tier\":\"dax\",\"status\":\"qualified\","
                   "\"qualified\":true,\"verification_failures\":0,"
                   "\"device\":" << capabilities.deviceOrdinal
                << ",\"window_bytes\":" << capabilities.windowBytes
                << ",\"mapped_device_address\":\"0x" << std::hex
                << reinterpret_cast<std::uintptr_t>(capabilities.mappedDeviceAddress)
                << std::dec << "\",\"host_registered\":true,"
                   "\"direct_device_visible\":true}\n";
    } else {
      std::cout << "cxl_dax_probe=passed"
                << " device=" << capabilities.deviceOrdinal
                << " window_bytes=" << capabilities.windowBytes
                << " mapped_device_address=0x" << std::hex
                << reinterpret_cast<std::uintptr_t>(
                       capabilities.mappedDeviceAddress)
                << std::dec << " host_registered=1 direct_device_visible=1\n";
    }
    return 0;
  } catch (const std::exception &error) {
    if (jsonOutput) {
      std::cout << "{\"schema\":1,\"classification\":\"nta-dax-qualification\","
                   "\"tier\":\"dax\",\"status\":\"failed\","
                   "\"qualified\":false,\"reason\":\"";
      for (const char character : std::string(error.what())) {
        if (character == '\\' || character == '"') {
          std::cout << '\\';
        }
        std::cout << character;
      }
      std::cout << "\"}\n";
    } else {
      std::cerr << "nta-cxl-dax-probe failed: " << error.what() << '\n';
    }
    return 1;
  }
}
