#pragma once

// Test and qualification frontends may discover an already-created devdax
// node, but the serving runtime still requires an explicit endpoint in its
// typed configuration.  Discovery never creates, deletes, or reconfigures a
// CXL region; it only removes the needless environment-variable step after a
// platform has already exposed a character device. The runtime repeats the
// ancestry check for explicit endpoints, so an environment variable cannot
// turn an arbitrary character device into a DAX qualification target.

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

#include <sys/stat.h>

namespace nta::qualification {

inline bool isCxlDaxNode(const std::string &name) {
  const std::filesystem::path classDevice =
      std::filesystem::path("/sys/class/dax") / name;
  const std::filesystem::path cxlDevices("/sys/bus/cxl/devices");
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::weakly_canonical(classDevice, error);
  if (error || resolved.empty()) {
    return false;
  }
  for (std::filesystem::path current = resolved; !current.empty();
       current = current.parent_path()) {
    const std::string component = current.filename().string();
    if (component.starts_with("region") &&
        std::filesystem::exists(cxlDevices / component, error) && !error) {
      return true;
    }
    if (current == current.parent_path()) {
      break;
    }
  }
  return false;
}

inline std::optional<std::string> discoverDaxEndpoint() {
  const std::filesystem::path devRoot("/dev");
  std::error_code error;
  if (!std::filesystem::is_directory(devRoot, error) || error) {
    return std::nullopt;
  }
  std::optional<std::string> endpoint;
  for (std::filesystem::directory_iterator it(devRoot, error), end;
       it != end && !error; it.increment(error)) {
    const std::filesystem::path path = it->path();
    const std::string name = path.filename().string();
    if (!name.starts_with("dax") || !isCxlDaxNode(name)) {
      continue;
    }
    struct stat status {};
    if (::stat(path.c_str(), &status) != 0 || !S_ISCHR(status.st_mode)) {
      continue;
    }
    endpoint = path.string();
    break;
  }
  return endpoint;
}

} // namespace nta::qualification
