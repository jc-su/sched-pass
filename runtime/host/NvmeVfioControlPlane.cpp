#include "NvmeControlPlane.h"

#include "nta/NvmeP2pUapi.h"
#include "nta/RuntimeABI.h"

#include <linux/iommufd.h>
#include <linux/vfio.h>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nta::detail {
namespace {

using Submission = abi::NvmeSubmission;
using Completion = abi::NvmeCompletion;

constexpr std::size_t NvmeRegisterCap = 0x0000;
constexpr std::size_t NvmeRegisterInterruptMaskSet = 0x000c;
constexpr std::size_t NvmeRegisterControllerConfig = 0x0014;
constexpr std::size_t NvmeRegisterControllerStatus = 0x001c;
constexpr std::size_t NvmeRegisterAdminQueueAttributes = 0x0024;
constexpr std::size_t NvmeRegisterAdminSubmissionQueue = 0x0028;
constexpr std::size_t NvmeRegisterAdminCompletionQueue = 0x0030;
constexpr std::size_t NvmeRegisterDoorbells = 0x1000;

constexpr std::uint32_t NvmeControllerEnable = 1U << 0U;
constexpr std::uint32_t NvmeControllerReady = 1U << 0U;
constexpr std::uint32_t NvmeControllerFatal = 1U << 1U;
constexpr std::uint32_t NvmeControllerPageShift = 7U;
constexpr std::uint32_t NvmeIoSubmissionEntrySize = 6U << 16U;
constexpr std::uint32_t NvmeIoCompletionEntrySize = 4U << 20U;
constexpr std::uint32_t NvmeAdminDepth = 32;
constexpr std::uint32_t NvmeIdentifyBytes = 4096;
constexpr std::uint32_t NvmeWriteProtectFeature = 0x84;
constexpr std::uint32_t NvmeBasicWriteProtect = 1;
constexpr std::uint32_t NvmeNumberOfQueuesFeature = 0x07;
constexpr std::uint32_t NvmeQueuePhysicallyContiguous = 1;
constexpr std::uint32_t NvmeAdminDeleteSq = 0x00;
constexpr std::uint32_t NvmeAdminCreateSq = 0x01;
constexpr std::uint32_t NvmeAdminDeleteCq = 0x04;
constexpr std::uint32_t NvmeAdminCreateCq = 0x05;
constexpr std::uint32_t NvmeAdminIdentify = 0x06;
constexpr std::uint32_t NvmeAdminSetFeatures = 0x09;
constexpr std::uint32_t NvmeAdminGetFeatures = 0x0a;
constexpr std::uint32_t NvmeIdentifyNamespace = 0x00;
constexpr std::uint32_t NvmeIdentifyController = 0x01;
constexpr std::uint32_t NvmeIdentifyActiveNamespaces = 0x02;
constexpr std::uint16_t PciCommandMemory = 0x0002;
constexpr std::uint16_t PciCommandMaster = 0x0004;
constexpr std::size_t PciCommandOffset = 0x04;
constexpr std::size_t PciClassDeviceOffset = 0x0a;
constexpr std::uint16_t PciClassNvme = 0x0108;
constexpr std::uint32_t MaximumQueueDepth = 4096;

[[noreturn]] void throwSystem(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

std::size_t roundUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0 ||
      value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    throw std::overflow_error("invalid VFIO DMA allocation size");
  }
  return (value + alignment - 1U) & ~(alignment - 1U);
}

template <typename T> T byteSwap(T value) {
  static_assert(std::is_unsigned_v<T>);
  if constexpr (sizeof(T) == sizeof(std::uint16_t)) {
    return static_cast<T>(__builtin_bswap16(static_cast<std::uint16_t>(value)));
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<T>(__builtin_bswap32(static_cast<std::uint32_t>(value)));
  } else {
    static_assert(sizeof(T) == sizeof(std::uint64_t));
    return static_cast<T>(__builtin_bswap64(static_cast<std::uint64_t>(value)));
  }
}

template <typename T> T loadLittle(const std::byte *data, std::size_t offset) {
  static_assert(std::is_unsigned_v<T>);
  T value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  if constexpr (std::endian::native == std::endian::big) {
    value = byteSwap(value);
  }
  return value;
}

std::uint32_t toLittle32(std::uint32_t value) {
  if constexpr (std::endian::native == std::endian::big) {
    return byteSwap(value);
  }
  return value;
}

std::uint64_t toLittle64(std::uint64_t value) {
  if constexpr (std::endian::native == std::endian::big) {
    return byteSwap(value);
  }
  return value;
}

std::uint32_t fromLittle32(std::uint32_t value) { return toLittle32(value); }

std::uint64_t fromLittle64(std::uint64_t value) { return toLittle64(value); }

class FileDescriptor {
public:
  FileDescriptor() = default;
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() { reset(); }
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  FileDescriptor(FileDescriptor &&other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other) {
      reset();
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return descriptor_; }
  void reset(int descriptor = -1) noexcept {
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
    }
    descriptor_ = descriptor;
  }

private:
  int descriptor_ = -1;
};

std::string parseBdf(const std::string &endpoint) {
  constexpr std::string_view prefix = "vfio:";
  if (!endpoint.starts_with(prefix)) {
    throw std::invalid_argument("VFIO endpoint must use vfio:DDDD:BB:SS.F");
  }
  const std::string bdf = endpoint.substr(prefix.size());
  const auto hex = [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
  };
  if (bdf.size() != 12 || bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.' ||
      !hex(bdf[0]) || !hex(bdf[1]) || !hex(bdf[2]) || !hex(bdf[3]) ||
      !hex(bdf[5]) || !hex(bdf[6]) || !hex(bdf[8]) || !hex(bdf[9]) ||
      bdf[11] < '0' || bdf[11] > '7') {
    throw std::invalid_argument("invalid PCI BDF in VFIO endpoint");
  }
  return bdf;
}

std::string resolveVfioCdev(const std::string &bdf) {
  const std::filesystem::path device =
      std::filesystem::path("/sys/bus/pci/devices") / bdf;
  if (!std::filesystem::exists(device)) {
    throw std::runtime_error("VFIO PCI device does not exist: " + bdf);
  }
  std::error_code error;
  const std::filesystem::path driver =
      std::filesystem::read_symlink(device / "driver", error);
  if (error || driver.filename() != "vfio-pci") {
    throw std::runtime_error("PCI device is not bound to vfio-pci; run the "
                             "VFIO preflight/bind tool");
  }
  const std::filesystem::path cdevDirectory = device / "vfio-dev";
  for (const auto &entry : std::filesystem::directory_iterator(cdevDirectory)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with("vfio")) {
      const std::filesystem::path node =
          std::filesystem::path("/dev/vfio/devices") / name;
      if (std::filesystem::exists(node)) {
        return node.string();
      }
    }
  }
  throw std::runtime_error(
      "vfio-pci cdev is unavailable; CONFIG_VFIO_DEVICE_CDEV is required");
}

std::string readTextFile(const std::filesystem::path &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throwSystem("open VFIO containment sysfs attribute");
  }
  std::array<char, 64> buffer{};
  const ssize_t bytes = ::read(fd, buffer.data(), buffer.size() - 1U);
  const int savedError = errno;
  (void)::close(fd);
  if (bytes < 0) {
    errno = savedError;
    throwSystem("read VFIO containment sysfs attribute");
  }
  return std::string(buffer.data(), static_cast<std::size_t>(bytes));
}

void validateIommuIsolation(const std::string &bdf) {
  const std::filesystem::path device =
      std::filesystem::path("/sys/bus/pci/devices") / bdf;
  std::error_code error;
  const std::filesystem::path group =
      std::filesystem::canonical(device / "iommu_group", error);
  if (error) {
    throw std::runtime_error("VFIO target has no IOMMU group");
  }
  std::size_t members = 0;
  bool targetFound = false;
  for (const auto &entry :
       std::filesystem::directory_iterator(group / "devices")) {
    ++members;
    targetFound = targetFound || entry.path().filename() == bdf;
  }
  if (members != 1 || !targetFound) {
    throw std::runtime_error("VFIO target is not alone in its IOMMU group");
  }
  const std::filesystem::path unsafe =
      "/sys/module/vfio/parameters/enable_unsafe_noiommu_mode";
  if (std::filesystem::exists(unsafe) &&
      readTextFile(unsafe).starts_with("Y")) {
    throw std::runtime_error("VFIO unsafe no-IOMMU mode is enabled");
  }
}

void setCommand(Submission &command, std::uint8_t opcode,
                std::uint16_t commandId) {
  command.dword[0] = toLittle32(static_cast<std::uint32_t>(opcode) |
                                (static_cast<std::uint32_t>(commandId) << 16U));
}

void setPrp1(Submission &command, std::uint64_t address) {
  const std::uint64_t little = toLittle64(address);
  std::memcpy(&command.dword[6], &little, sizeof(little));
}

class VfioNvmeControlPlane final : public NvmeControlPlane,
                                   public NvmeMappingBackend {
public:
  explicit VfioNvmeControlPlane(const NvmeTransportOptions &options)
      : bdf_(parseBdf(options.endpoint)), namespaceId_(options.namespaceId),
        requestedQueueDepth_(options.queueDepth),
        adminTimeout_(options.adminTimeoutMs),
        mediaPolicy_(options.mediaPolicy) {
    if (namespaceId_ == 0 || requestedQueueDepth_ < 2 ||
        requestedQueueDepth_ > MaximumQueueDepth ||
        options.adminTimeoutMs == 0) {
      throw std::invalid_argument(
          "VFIO namespace, queue depth, and admin timeout must be valid");
    }
    const long systemPage = ::sysconf(_SC_PAGESIZE);
    if (systemPage != static_cast<long>(NvmeIdentifyBytes)) {
      throw std::runtime_error(
          "VFIO NVMe requires 4 KiB host pages to isolate doorbells");
    }
    pageSize_ = static_cast<std::size_t>(systemPage);

    try {
      openAndAttach();
      mapBarAndReset();
      inspectCapabilities();
      allocateAdminMemory();
      enableController();
      identifyControllerAndNamespace();
      protectActiveNamespaces();
      configureIoQueues();
      allocateAndCreateIoQueue();
      publishResources(options.deviceOrdinal);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~VfioNvmeControlPlane() override { cleanup(); }

  [[nodiscard]] const NvmeQueueResources &resources() const noexcept override {
    return resources_;
  }

  [[nodiscard]] NvmeMappingBackend &mappingBackend() noexcept override {
    return *this;
  }

  [[nodiscard]] NvmeMapping mapHost(void *address, std::size_t bytes) override {
    std::scoped_lock lock(mutex_);
    if (quiesced_ || fatal_) {
      throw std::runtime_error("VFIO NVMe queue is not accepting DMA mappings");
    }
    // Peer PTEs are installed directly in the attached IOMMU domain and are
    // intentionally outside IOMMUFD's userspace IOVA allocator. All ordinary
    // IOAS mappings must therefore be complete before the first peer map so a
    // later allocator choice cannot collide with an HBM IOVA.
    if (!peerMappings_.empty()) {
      throw std::runtime_error(
          "VFIO host mappings must precede every HBM peer mapping");
    }
    if (reinterpret_cast<std::uintptr_t>(address) % pageSize_ != 0 ||
        bytes == 0 || bytes % pageSize_ != 0) {
      throw std::invalid_argument("VFIO DMA mapping must be page aligned");
    }
    return publishHostMapping(mapIoas(address, bytes), bytes);
  }

  [[nodiscard]] NvmeMapping mapHbm(std::uint64_t gpuAddress,
                                   std::size_t bytes) override {
    std::scoped_lock lock(mutex_);
    if (quiesced_ || fatal_) {
      throw std::runtime_error(
          "VFIO NVMe queue is not accepting HBM peer mappings");
    }
    constexpr std::size_t peerAlignment = 64U * 1024U;
    if (gpuAddress == 0 || gpuAddress % peerAlignment != 0 || bytes == 0 ||
        bytes % peerAlignment != 0 || bytes % pageSize_ != 0) {
      throw std::invalid_argument(
          "NVMe HBM peer mapping must be 64 KiB aligned");
    }
    return mapNvidiaPeerPages(gpuAddress, bytes);
  }

  void release(NvmeMappingToken token) noexcept override {
    if (!token) {
      return;
    }
    std::scoped_lock lock(mutex_);
    if (token.kind == NvmeMappingToken::Kind::HostIoas) {
      if (iommufd_.get() < 0) {
        return;
      }
      const auto mapping = dmaMappings_.find(token.value);
      if (mapping == dmaMappings_.end()) {
        return;
      }
      unmapIoas(mapping->first, mapping->second);
      dmaMappings_.erase(mapping);
    } else if (token.kind == NvmeMappingToken::Kind::NvidiaPeerPages) {
      unmapNvidiaPeerPages(token.value);
    }
  }

  void quiesce() noexcept override {
    std::scoped_lock lock(mutex_);
    quiesceLocked();
  }

private:
  NvmeMapping publishHostMapping(std::uint64_t iova, std::size_t bytes) {
    if (iova == 0 || bytes == 0 || bytes % pageSize_ != 0 ||
        bytes > std::numeric_limits<std::uint64_t>::max() - iova) {
      if (iova != 0) {
        unmapIoas(iova, bytes);
      }
      throw std::runtime_error("VFIO returned an invalid DMA mapping range");
    }
    try {
      const auto [mapping, inserted] = dmaMappings_.emplace(iova, bytes);
      if (!inserted) {
        throw std::runtime_error("VFIO returned a duplicate DMA IOVA");
      }
      (void)mapping;
      std::vector<std::uint64_t> pages;
      pages.reserve(bytes / pageSize_);
      for (std::size_t offset = 0; offset < bytes; offset += pageSize_) {
        pages.push_back(iova + offset);
      }
      return makeMapping({NvmeMappingToken::Kind::HostIoas, iova},
                         std::move(pages));
    } catch (...) {
      if (iova != 0) {
        unmapIoas(iova, bytes);
      }
      dmaMappings_.erase(iova);
      throw;
    }
  }

  NvmeMapping mapNvidiaPeerPages(std::uint64_t gpuAddress, std::size_t bytes) {
    if (peerMapper_.get() < 0) {
      peerMapper_.reset(::open(NTA_NVME_P2P_DEVICE_PATH, O_RDWR | O_CLOEXEC));
      if (peerMapper_.get() < 0) {
        throwSystem("open " NTA_NVME_P2P_DEVICE_PATH);
      }
    }
    const std::size_t maximumEntries = bytes / pageSize_;
    if (maximumEntries == 0 ||
        maximumEntries > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("NVMe HBM peer page count overflows UAPI");
    }
    std::vector<std::uint64_t> nativeAddresses(maximumEntries);
    nta_nvme_p2p_map request{};
    request.size = sizeof(request);
    request.abi_version = NTA_NVME_P2P_ABI_VERSION;
    request.gpu_address = gpuAddress;
    request.bytes = bytes;
    request.pci_domain =
        static_cast<std::uint32_t>(std::stoul(bdf_.substr(0, 4), nullptr, 16));
    request.pci_bus =
        static_cast<std::uint32_t>(std::stoul(bdf_.substr(5, 2), nullptr, 16));
    request.pci_device =
        static_cast<std::uint32_t>(std::stoul(bdf_.substr(8, 2), nullptr, 16));
    request.pci_function =
        static_cast<std::uint32_t>(std::stoul(bdf_.substr(11, 1), nullptr, 16));
    request.dma_addresses =
        reinterpret_cast<std::uint64_t>(nativeAddresses.data());
    request.dma_capacity = static_cast<std::uint32_t>(maximumEntries);
    if (::ioctl(peerMapper_.get(), NTA_NVME_P2P_IOCTL_MAP, &request) != 0) {
      throwSystem("NTA_NVME_P2P_IOCTL_MAP");
    }
    const auto rollback = [&]() noexcept {
      nta_nvme_p2p_unmap unmapRequest{sizeof(unmapRequest),
                                      NTA_NVME_P2P_ABI_VERSION, request.handle};
      (void)::ioctl(peerMapper_.get(), NTA_NVME_P2P_IOCTL_UNMAP, &unmapRequest);
    };
    if (request.handle == 0 || request.entry_count == 0 ||
        request.entry_count > maximumEntries || request.page_size < pageSize_ ||
        request.page_size % pageSize_ != 0 ||
        static_cast<std::uint64_t>(request.entry_count) * request.page_size !=
            bytes) {
      rollback();
      throw std::runtime_error(
          "NVIDIA peer mapper returned an invalid DMA-page vector");
    }
    try {
      std::vector<std::uint64_t> pages;
      pages.reserve(bytes / pageSize_);
      for (std::uint32_t index = 0; index < request.entry_count; ++index) {
        const std::uint64_t base = nativeAddresses[index];
        if (base == 0 || base % pageSize_ != 0 ||
            base > std::numeric_limits<std::uint64_t>::max() -
                       (request.page_size - pageSize_)) {
          throw std::runtime_error(
              "NVIDIA peer mapper returned a misaligned DMA address");
        }
        for (std::uint32_t offset = 0; offset < request.page_size;
             offset += static_cast<std::uint32_t>(pageSize_)) {
          pages.push_back(base + offset);
        }
      }
      if (pages.size() != bytes / pageSize_) {
        throw std::runtime_error(
            "expanded NVIDIA peer vector does not cover the HBM range");
      }
      const auto [ignored, inserted] = peerMappings_.insert(request.handle);
      (void)ignored;
      if (!inserted) {
        throw std::runtime_error(
            "NVIDIA peer mapper returned a duplicate handle");
      }
      return makeMapping(
          {NvmeMappingToken::Kind::NvidiaPeerPages, request.handle},
          std::move(pages));
    } catch (...) {
      rollback();
      throw;
    }
  }

  void unmapNvidiaPeerPages(std::uint64_t handle) noexcept {
    if (peerMapper_.get() < 0 || peerMappings_.erase(handle) == 0) {
      return;
    }
    nta_nvme_p2p_unmap request{sizeof(request), NTA_NVME_P2P_ABI_VERSION,
                               handle};
    (void)::ioctl(peerMapper_.get(), NTA_NVME_P2P_IOCTL_UNMAP, &request);
  }

  struct DmaRegion {
    void *host = nullptr;
    std::size_t bytes = 0;
    std::uint64_t iova = 0;
  };

  void openAndAttach() {
    validateIommuIsolation(bdf_);
    const std::string cdev = resolveVfioCdev(bdf_);
    iommufd_.reset(::open("/dev/iommu", O_RDWR | O_CLOEXEC));
    if (iommufd_.get() < 0) {
      throwSystem("open /dev/iommu");
    }
    vfioDevice_.reset(::open(cdev.c_str(), O_RDWR | O_CLOEXEC));
    if (vfioDevice_.get() < 0) {
      throwSystem("open VFIO PCI cdev");
    }

    vfio_device_bind_iommufd bind{};
    bind.argsz = sizeof(bind);
    bind.iommufd = iommufd_.get();
    if (::ioctl(vfioDevice_.get(), VFIO_DEVICE_BIND_IOMMUFD, &bind) != 0) {
      throwSystem("VFIO_DEVICE_BIND_IOMMUFD");
    }

    iommu_ioas_alloc allocation{};
    allocation.size = sizeof(allocation);
    if (::ioctl(iommufd_.get(), IOMMU_IOAS_ALLOC, &allocation) != 0) {
      throwSystem("IOMMU_IOAS_ALLOC");
    }
    ioasId_ = allocation.out_ioas_id;

    vfio_device_attach_iommufd_pt attach{};
    attach.argsz = sizeof(attach);
    attach.pt_id = ioasId_;
    if (::ioctl(vfioDevice_.get(), VFIO_DEVICE_ATTACH_IOMMUFD_PT, &attach) !=
        0) {
      throwSystem("VFIO_DEVICE_ATTACH_IOMMUFD_PT");
    }
    attached_ = true;

    vfio_device_info deviceInfo{};
    deviceInfo.argsz = sizeof(deviceInfo);
    if (::ioctl(vfioDevice_.get(), VFIO_DEVICE_GET_INFO, &deviceInfo) != 0) {
      throwSystem("VFIO_DEVICE_GET_INFO");
    }
    if ((deviceInfo.flags & VFIO_DEVICE_FLAGS_PCI) == 0 ||
        (deviceInfo.flags & VFIO_DEVICE_FLAGS_RESET) == 0 ||
        deviceInfo.num_regions <= VFIO_PCI_CONFIG_REGION_INDEX) {
      throw std::runtime_error(
          "VFIO device lacks PCI regions or reset support");
    }
    barRegion_ = getRegion(VFIO_PCI_BAR0_REGION_INDEX);
    configRegion_ = getRegion(VFIO_PCI_CONFIG_REGION_INDEX);
    validatePciClass();
  }

  void validatePciClass() const {
    std::uint16_t classDevice = 0;
    const off_t offset =
        static_cast<off_t>(configRegion_.offset + PciClassDeviceOffset);
    if (::pread(vfioDevice_.get(), &classDevice, sizeof(classDevice), offset) !=
        static_cast<ssize_t>(sizeof(classDevice))) {
      throwSystem("pread VFIO PCI class");
    }
    if constexpr (std::endian::native == std::endian::big) {
      classDevice = byteSwap(classDevice);
    }
    if (classDevice != PciClassNvme) {
      throw std::runtime_error("VFIO target is not an NVMe controller");
    }
  }

  vfio_region_info getRegion(std::uint32_t index) const {
    vfio_region_info region{};
    region.argsz = sizeof(region);
    region.index = index;
    if (::ioctl(vfioDevice_.get(), VFIO_DEVICE_GET_REGION_INFO, &region) != 0) {
      throwSystem("VFIO_DEVICE_GET_REGION_INFO");
    }
    return region;
  }

  void mapBarAndReset() {
    constexpr std::uint32_t required = VFIO_REGION_INFO_FLAG_READ |
                                       VFIO_REGION_INFO_FLAG_WRITE |
                                       VFIO_REGION_INFO_FLAG_MMAP;
    if ((barRegion_.flags & required) != required ||
        barRegion_.size < NvmeRegisterDoorbells + pageSize_ ||
        configRegion_.size < PciCommandOffset + sizeof(std::uint16_t)) {
      throw std::runtime_error("VFIO BAR0/config regions are not usable");
    }
    bar_ = ::mmap(nullptr, pageSize_, PROT_READ | PROT_WRITE, MAP_SHARED,
                  vfioDevice_.get(), static_cast<off_t>(barRegion_.offset));
    if (bar_ == MAP_FAILED) {
      bar_ = nullptr;
      throwSystem("mmap VFIO NVMe control page");
    }
    doorbellBar_ =
        ::mmap(nullptr, pageSize_, PROT_READ | PROT_WRITE, MAP_SHARED,
               vfioDevice_.get(),
               static_cast<off_t>(barRegion_.offset + NvmeRegisterDoorbells));
    if (doorbellBar_ == MAP_FAILED) {
      doorbellBar_ = nullptr;
      throwSystem("mmap VFIO NVMe doorbell page");
    }
    if (::ioctl(vfioDevice_.get(), VFIO_DEVICE_RESET) != 0) {
      throwSystem("VFIO_DEVICE_RESET");
    }
    resetPerformed_ = true;
    setBusMaster(true);
  }

  void inspectCapabilities() {
    cap_ = readMmio64(NvmeRegisterCap);
    const std::uint32_t minimumPageShift =
        12U + static_cast<std::uint32_t>((cap_ >> 48U) & 0xfU);
    const std::uint32_t maximumPageShift =
        12U + static_cast<std::uint32_t>((cap_ >> 52U) & 0xfU);
    const std::uint32_t hostPageShift = std::countr_zero(pageSize_);
    if (hostPageShift < minimumPageShift || hostPageShift > maximumPageShift) {
      throw std::runtime_error(
          "NVMe controller does not support host page size");
    }
    const std::uint32_t maximumDepth =
        static_cast<std::uint32_t>(cap_ & 0xffffU) + 1U;
    if (requestedQueueDepth_ > maximumDepth) {
      throw std::invalid_argument("requested queue depth exceeds CAP.MQES");
    }
    if (((cap_ >> 37U) & 1U) == 0) {
      throw std::runtime_error("NVMe controller lacks the NVM command set");
    }
    doorbellStride_ = 4U << static_cast<std::uint32_t>((cap_ >> 32U) & 0xfU);
    if (3ULL * doorbellStride_ + sizeof(std::uint32_t) > pageSize_) {
      throw std::runtime_error("I/O queue doorbells escape isolated BAR page");
    }
    const std::uint32_t timeoutUnits =
        static_cast<std::uint32_t>((cap_ >> 24U) & 0xffU);
    controllerTimeout_ = std::chrono::milliseconds(
        std::max<std::uint32_t>(500, timeoutUnits * 500));
  }

  DmaRegion allocateDma(std::size_t bytes) {
    DmaRegion region;
    region.bytes = roundUp(bytes, pageSize_);
    region.host = ::mmap(nullptr, region.bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (region.host == MAP_FAILED) {
      region.host = nullptr;
      throwSystem("mmap VFIO DMA memory");
    }
    std::memset(region.host, 0, region.bytes);
    try {
      region.iova = mapIoas(region.host, region.bytes);
    } catch (...) {
      (void)::munmap(region.host, region.bytes);
      throw;
    }
    return region;
  }

  void releaseDma(DmaRegion &region) noexcept {
    if (region.iova != 0) {
      unmapIoas(region.iova, region.bytes);
      region.iova = 0;
    }
    if (region.host != nullptr) {
      (void)::munmap(region.host, region.bytes);
      region.host = nullptr;
    }
    region.bytes = 0;
  }

  std::uint64_t mapIoas(void *address, std::size_t bytes) {
    iommu_ioas_map request{};
    request.size = sizeof(request);
    request.flags = IOMMU_IOAS_MAP_READABLE | IOMMU_IOAS_MAP_WRITEABLE;
    request.ioas_id = ioasId_;
    request.user_va = reinterpret_cast<std::uint64_t>(address);
    request.length = bytes;
    if (::ioctl(iommufd_.get(), IOMMU_IOAS_MAP, &request) != 0) {
      throwSystem("IOMMU_IOAS_MAP");
    }
    if (request.iova == 0 || request.iova % pageSize_ != 0) {
      if (request.iova != 0) {
        unmapIoas(request.iova, bytes);
      }
      throw std::runtime_error("IOMMUFD returned an invalid IOVA");
    }
    return request.iova;
  }

  void unmapIoas(std::uint64_t iova, std::size_t bytes) noexcept {
    if (iommufd_.get() < 0 || ioasId_ == 0 || bytes == 0) {
      return;
    }
    iommu_ioas_unmap request{};
    request.size = sizeof(request);
    request.ioas_id = ioasId_;
    request.iova = iova;
    request.length = bytes;
    (void)::ioctl(iommufd_.get(), IOMMU_IOAS_UNMAP, &request);
  }

  void allocateAdminMemory() {
    adminSqOffset_ = 0;
    adminCqOffset_ = roundUp(NvmeAdminDepth * sizeof(Submission), pageSize_);
    identifyOffset_ = adminCqOffset_ +
                      roundUp(NvmeAdminDepth * sizeof(Completion), pageSize_);
    adminMemory_ = allocateDma(identifyOffset_ + NvmeIdentifyBytes);
    adminSq_ = reinterpret_cast<Submission *>(
        static_cast<std::byte *>(adminMemory_.host) + adminSqOffset_);
    adminCq_ = reinterpret_cast<Completion *>(
        static_cast<std::byte *>(adminMemory_.host) + adminCqOffset_);
    identify_ = static_cast<std::byte *>(adminMemory_.host) + identifyOffset_;
  }

  void enableController() {
    const std::uint32_t config = readMmio32(NvmeRegisterControllerConfig);
    if ((config & NvmeControllerEnable) != 0) {
      writeMmio32(NvmeRegisterControllerConfig, config & ~NvmeControllerEnable);
      waitReady(false);
    }
    std::memset(adminSq_, 0, NvmeAdminDepth * sizeof(Submission));
    std::memset(adminCq_, 0, NvmeAdminDepth * sizeof(Completion));
    adminTail_ = 0;
    adminHead_ = 0;
    adminPhase_ = 1;
    adminCid_ = 1;
    writeMmio32(NvmeRegisterInterruptMaskSet, 0xffffffffU);
    writeMmio32(NvmeRegisterAdminQueueAttributes,
                ((NvmeAdminDepth - 1U) << 16U) | (NvmeAdminDepth - 1U));
    writeMmio64(NvmeRegisterAdminSubmissionQueue,
                adminMemory_.iova + adminSqOffset_);
    writeMmio64(NvmeRegisterAdminCompletionQueue,
                adminMemory_.iova + adminCqOffset_);
    const std::uint32_t pageShift = std::countr_zero(pageSize_) - 12U;
    writeMmio32(NvmeRegisterControllerConfig,
                NvmeControllerEnable | (pageShift << NvmeControllerPageShift) |
                    NvmeIoSubmissionEntrySize | NvmeIoCompletionEntrySize);
    waitReady(true);
    controllerEnabled_ = true;
  }

  std::uint32_t adminCommand(const Submission &source) {
    Submission command = source;
    if (fatal_ || !controllerEnabled_) {
      throw std::runtime_error("NVMe controller is unavailable");
    }
    const std::uint16_t cid = adminCid_++;
    const std::uint32_t first = fromLittle32(command.dword[0]);
    setCommand(command, static_cast<std::uint8_t>(first & 0xffU), cid);
    std::memcpy(&adminSq_[adminTail_], &command, sizeof(command));
    adminTail_ = (adminTail_ + 1U) % NvmeAdminDepth;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    writeDoorbell(0, false, adminTail_);

    const auto deadline = std::chrono::steady_clock::now() + adminTimeout_;
    Completion *completion = &adminCq_[adminHead_];
    const auto *completionDwordAddress =
        reinterpret_cast<volatile std::uint32_t *>(&completion->dword[3]);
    std::uint32_t completionDword = 0;
    while (true) {
      completionDword = fromLittle32(*completionDwordAddress);
      const std::uint16_t status =
          static_cast<std::uint16_t>(completionDword >> 16U);
      if ((status & 1U) == adminPhase_) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        fatal_ = true;
        throw std::runtime_error(
            "NVMe admin command timed out; controller poisoned");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const std::uint16_t completionCid =
        static_cast<std::uint16_t>(completionDword & 0xffffU);
    const std::uint16_t status =
        static_cast<std::uint16_t>(completionDword >> 16U);
    if (completionCid != cid) {
      fatal_ = true;
      throw std::runtime_error("NVMe admin completion command ID mismatch");
    }
    const std::uint32_t result = fromLittle32(completion->dword[0]);
    adminHead_++;
    if (adminHead_ == NvmeAdminDepth) {
      adminHead_ = 0;
      adminPhase_ ^= 1U;
    }
    writeDoorbell(0, true, adminHead_);
    if ((status >> 1U) != 0) {
      throw std::runtime_error("NVMe admin command failed with status " +
                               std::to_string(status >> 1U));
    }
    return result;
  }

  void identify(std::uint32_t cns, std::uint32_t nsid) {
    std::memset(identify_, 0, NvmeIdentifyBytes);
    Submission command{};
    setCommand(command, NvmeAdminIdentify, 0);
    command.dword[1] = toLittle32(nsid);
    setPrp1(command, adminMemory_.iova + identifyOffset_);
    command.dword[10] = toLittle32(cns);
    (void)adminCommand(command);
  }

  void identifyControllerAndNamespace() {
    identify(NvmeIdentifyController, 0);
    namespaceCount_ = loadLittle<std::uint32_t>(identify_, 516);
    const std::uint8_t mdts = static_cast<std::uint8_t>(identify_[77]);
    const std::uint8_t namespaceWriteProtect =
        static_cast<std::uint8_t>(identify_[531]);
    if (namespaceId_ > namespaceCount_) {
      throw std::runtime_error("NVMe target namespace is absent");
    }
    writeProtectSupported_ = (namespaceWriteProtect & 1U) != 0;
    const std::uint64_t prpCapacity =
        pageSize_ * (pageSize_ / sizeof(std::uint64_t));
    std::uint64_t mdtsBytes = prpCapacity;
    if (mdts != 0 && mdts < 9) {
      const std::uint64_t reported = pageSize_ << mdts;
      mdtsBytes = std::min(reported, prpCapacity);
    }
    maxTransferBytes_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        mdtsBytes, std::numeric_limits<std::uint32_t>::max()));

    identify(NvmeIdentifyNamespace, namespaceId_);
    namespaceBlocks_ = loadLittle<std::uint64_t>(identify_, 0);
    const std::uint8_t formats = static_cast<std::uint8_t>(identify_[25]);
    const std::uint8_t flbas = static_cast<std::uint8_t>(identify_[26]);
    const std::uint8_t format =
        static_cast<std::uint8_t>((flbas & 0x0fU) | ((flbas & 0x60U) >> 1U));
    const std::uint8_t protection = static_cast<std::uint8_t>(identify_[29]);
    if (namespaceBlocks_ == 0 || format > formats || (protection & 0x7U) != 0) {
      throw std::runtime_error("NVMe namespace format is unsupported");
    }
    const std::size_t formatOffset = 128U + 4U * format;
    const std::uint16_t metadata =
        loadLittle<std::uint16_t>(identify_, formatOffset);
    lbaShift_ = static_cast<std::uint8_t>(identify_[formatOffset + 2U]);
    if (metadata != 0 || lbaShift_ < 9 ||
        lbaShift_ > std::countr_zero(pageSize_)) {
      throw std::runtime_error(
          "NVMe namespace metadata or LBA size is unsupported");
    }
    if (namespaceBlocks_ >
        (std::numeric_limits<std::uint64_t>::max() >> lbaShift_)) {
      throw std::runtime_error("NVMe namespace byte size overflows uint64_t");
    }
  }

  void setNamespaceReadOnly(std::uint32_t nsid) {
    Submission set{};
    setCommand(set, NvmeAdminSetFeatures, 0);
    set.dword[1] = toLittle32(nsid);
    set.dword[10] = toLittle32(NvmeWriteProtectFeature);
    set.dword[11] = toLittle32(NvmeBasicWriteProtect);
    (void)adminCommand(set);

    Submission get{};
    setCommand(get, NvmeAdminGetFeatures, 0);
    get.dword[1] = toLittle32(nsid);
    get.dword[10] = toLittle32(NvmeWriteProtectFeature);
    if ((adminCommand(get) & 0x7U) != NvmeBasicWriteProtect) {
      throw std::runtime_error("NVMe namespace write protection did not latch");
    }
  }

  void protectActiveNamespaces() {
    // The trusted-read-only-code contract is deliberately a software-only
    // contract.  Do not issue Set Features (FID 84h) in that mode: changing
    // the namespace write-protection state is an externally visible media
    // mutation and may persist across a controller reset.  The benchmark and
    // runtime must remain read-only by construction instead.
    if (mediaPolicy_ == NvmeMediaPolicy::TrustReadOnlyDeviceCode) {
      namespaceReadOnly_ = false;
      return;
    }
    if (!writeProtectSupported_) {
      namespaceReadOnly_ = false;
      return;
    }
    std::uint32_t cursor = 0;
    std::uint32_t protectedCount = 0;
    bool targetFound = false;
    while (protectedCount < namespaceCount_) {
      identify(NvmeIdentifyActiveNamespaces, cursor);
      bool any = false;
      for (std::size_t offset = 0; offset < NvmeIdentifyBytes;
           offset += sizeof(std::uint32_t)) {
        const std::uint32_t nsid = loadLittle<std::uint32_t>(identify_, offset);
        if (nsid == 0) {
          namespaceReadOnly_ = targetFound;
          if (!targetFound) {
            throw std::runtime_error("target NVMe namespace is not active");
          }
          return;
        }
        if (nsid <= cursor) {
          throw std::runtime_error("invalid active namespace list ordering");
        }
        setNamespaceReadOnly(nsid);
        targetFound = targetFound || nsid == namespaceId_;
        cursor = nsid;
        protectedCount++;
        any = true;
      }
      if (!any) {
        break;
      }
    }
    namespaceReadOnly_ = targetFound;
    if (!targetFound) {
      throw std::runtime_error("target NVMe namespace is not active");
    }
  }

  void configureIoQueues() {
    Submission command{};
    setCommand(command, NvmeAdminSetFeatures, 0);
    command.dword[10] = toLittle32(NvmeNumberOfQueuesFeature);
    command.dword[11] = 0;
    const std::uint32_t result = adminCommand(command);
    const std::uint32_t submissionQueues = (result & 0xffffU) + 1U;
    const std::uint32_t completionQueues = (result >> 16U) + 1U;
    queueCount_ = std::min(submissionQueues, completionQueues);
    if (queueCount_ == 0) {
      throw std::runtime_error("NVMe controller allocated no I/O queues");
    }
  }

  void allocateAndCreateIoQueue() {
    controlOffset_ = 0;
    sqOffset_ = pageSize_;
    cqOffset_ = sqOffset_ +
                roundUp(requestedQueueDepth_ * sizeof(Submission), pageSize_);
    prpOffset_ = cqOffset_ +
                 roundUp(requestedQueueDepth_ * sizeof(Completion), pageSize_);
    queueMemory_ = allocateDma(prpOffset_ + requestedQueueDepth_ * pageSize_);
    auto *control = reinterpret_cast<abi::NvmeQueueControl *>(
        static_cast<std::byte *>(queueMemory_.host) + controlOffset_);
    control->magic = abi::NvmeQueueControlMagic;
    control->abiVersion = abi::NvmeQueueAbiVersion;
    control->state = static_cast<std::uint32_t>(abi::NvmeQueueState::Offline);
    control->generation = generation_;
    control->queueId = queueId_;

    createIoQueuePair();
    validateCpuIoRead();
    deleteQueue(NvmeAdminDeleteSq, queueId_);
    sqLive_ = false;
    deleteQueue(NvmeAdminDeleteCq, queueId_);
    cqLive_ = false;
    std::memset(static_cast<std::byte *>(queueMemory_.host) + sqOffset_, 0,
                queueMemory_.bytes - sqOffset_);
    createIoQueuePair();
    std::atomic_thread_fence(std::memory_order_release);
    control->state = static_cast<std::uint32_t>(abi::NvmeQueueState::Online);
  }

  void createIoQueuePair() {
    Submission createCq{};
    setCommand(createCq, NvmeAdminCreateCq, 0);
    setPrp1(createCq, queueMemory_.iova + cqOffset_);
    createCq.dword[10] =
        toLittle32(queueId_ | ((requestedQueueDepth_ - 1U) << 16U));
    createCq.dword[11] = toLittle32(NvmeQueuePhysicallyContiguous);
    (void)adminCommand(createCq);
    cqLive_ = true;

    try {
      Submission createSq{};
      setCommand(createSq, NvmeAdminCreateSq, 0);
      setPrp1(createSq, queueMemory_.iova + sqOffset_);
      createSq.dword[10] =
          toLittle32(queueId_ | ((requestedQueueDepth_ - 1U) << 16U));
      createSq.dword[11] =
          toLittle32(NvmeQueuePhysicallyContiguous | (queueId_ << 16U));
      (void)adminCommand(createSq);
      sqLive_ = true;
    } catch (...) {
      deleteQueue(NvmeAdminDeleteCq, queueId_);
      cqLive_ = false;
      throw;
    }
  }

  void validateCpuIoRead() {
    constexpr std::uint16_t commandId = 0x7ffe;
    auto *submission = reinterpret_cast<Submission *>(
        static_cast<std::byte *>(queueMemory_.host) + sqOffset_);
    auto *completion = reinterpret_cast<Completion *>(
        static_cast<std::byte *>(queueMemory_.host) + cqOffset_);
    std::memset(identify_, 0, NvmeIdentifyBytes);
    std::memset(submission, 0, sizeof(*submission));
    std::memset(completion, 0, sizeof(*completion));
    setCommand(*submission, 0x02, commandId);
    submission->dword[1] = toLittle32(namespaceId_);
    setPrp1(*submission, adminMemory_.iova + identifyOffset_);
    submission->dword[10] = 0;
    submission->dword[11] = 0;
    submission->dword[12] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    writeDoorbell(queueId_, false, 1);

    const auto deadline = std::chrono::steady_clock::now() + adminTimeout_;
    const auto *completionDwordAddress =
        reinterpret_cast<volatile std::uint32_t *>(&completion->dword[3]);
    std::uint32_t completionDword = 0;
    while (true) {
      completionDword = fromLittle32(*completionDwordAddress);
      const std::uint16_t status =
          static_cast<std::uint16_t>(completionDword >> 16U);
      if ((status & 1U) == 1U) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        fatal_ = true;
        throw std::runtime_error("NVMe CPU I/O queue self-test timed out");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const std::uint16_t completionCid =
        static_cast<std::uint16_t>(completionDword & 0xffffU);
    const std::uint16_t status =
        static_cast<std::uint16_t>(completionDword >> 16U);
    if (completionCid != commandId || (status >> 1U) != 0) {
      fatal_ = true;
      throw std::runtime_error("NVMe CPU I/O queue self-test failed");
    }
    writeDoorbell(queueId_, true, 1);
  }

  void publishResources(int deviceOrdinal) {
    resources_.capabilities.queueDepth = requestedQueueDepth_;
    resources_.capabilities.controllerPageSize =
        static_cast<std::uint32_t>(pageSize_);
    resources_.capabilities.lbaSize = 1U << lbaShift_;
    resources_.capabilities.maxTransferBytes = maxTransferBytes_;
    resources_.capabilities.namespaceBytes = namespaceBlocks_ << lbaShift_;
    resources_.capabilities.queueId = queueId_;
    // queueCount_ is the controller's negotiated availability. This control
    // plane currently creates one live GPU-owned pair; report active queues,
    // not an availability value that callers could mistake for parallelism.
    resources_.capabilities.queueCount = 1;
    resources_.capabilities.deviceOrdinal = deviceOrdinal;
    resources_.capabilities.supportsHbmPeerDma = false;
    resources_.capabilities.hbmMappingBackend =
        NvmeHbmMappingBackend::Unavailable;
    resources_.capabilities.translatedIommu = true;
    resources_.capabilities.namespaceReadOnly = namespaceReadOnly_;
    resources_.capabilities.gpuDoorbellMappingValidated = false;
    resources_.queueHost = queueMemory_.host;
    resources_.queueBytes = queueMemory_.bytes;
    resources_.controlOffset = controlOffset_;
    resources_.sqOffset = sqOffset_;
    resources_.cqOffset = cqOffset_;
    resources_.prpOffset = prpOffset_;
    resources_.prpDmaAddress = queueMemory_.iova + prpOffset_;
    resources_.doorbellHost = doorbellBar_;
    resources_.doorbellBytes = pageSize_;
    resources_.sqDoorbellOffset = 2U * queueId_ * doorbellStride_;
    resources_.cqDoorbellOffset = (2U * queueId_ + 1U) * doorbellStride_;
    resources_.generation = generation_;
    resources_.queueIsIoMemory = false;
  }

  void deleteQueue(std::uint8_t opcode, std::uint32_t qid) {
    Submission command{};
    setCommand(command, opcode, 0);
    command.dword[10] = toLittle32(qid);
    (void)adminCommand(command);
  }

  void quiesceLocked() noexcept {
    if (quiesced_) {
      return;
    }
    quiesced_ = true;
    if (queueMemory_.host != nullptr) {
      auto *control = reinterpret_cast<abi::NvmeQueueControl *>(
          static_cast<std::byte *>(queueMemory_.host) + controlOffset_);
      control->state =
          static_cast<std::uint32_t>(abi::NvmeQueueState::Quiesced);
    }
    try {
      if (!fatal_ && sqLive_) {
        deleteQueue(NvmeAdminDeleteSq, queueId_);
        sqLive_ = false;
      }
      if (!fatal_ && cqLive_) {
        deleteQueue(NvmeAdminDeleteCq, queueId_);
        cqLive_ = false;
      }
    } catch (...) {
      fatal_ = true;
    }
    if (fatal_ && vfioDevice_.get() >= 0) {
      setBusMasterNoexcept(false);
      (void)::ioctl(vfioDevice_.get(), VFIO_DEVICE_RESET);
      controllerEnabled_ = false;
      sqLive_ = false;
      cqLive_ = false;
      if (queueMemory_.host != nullptr) {
        auto *control = reinterpret_cast<abi::NvmeQueueControl *>(
            static_cast<std::byte *>(queueMemory_.host) + controlOffset_);
        control->state = static_cast<std::uint32_t>(abi::NvmeQueueState::Fatal);
      }
    }
  }

  void cleanup() noexcept {
    {
      std::scoped_lock lock(mutex_);
      quiesceLocked();
      if (controllerEnabled_ && bar_ != nullptr) {
        const std::uint32_t config = readMmio32(NvmeRegisterControllerConfig);
        writeMmio32(NvmeRegisterControllerConfig,
                    config & ~NvmeControllerEnable);
        waitReadyNoexcept(false);
        controllerEnabled_ = false;
      }
      setBusMasterNoexcept(false);
      if (vfioDevice_.get() >= 0 && resetPerformed_) {
        (void)::ioctl(vfioDevice_.get(), VFIO_DEVICE_RESET);
      }
      for (const std::uint64_t handle : peerMappings_) {
        nta_nvme_p2p_unmap request{sizeof(request), NTA_NVME_P2P_ABI_VERSION,
                                   handle};
        (void)::ioctl(peerMapper_.get(), NTA_NVME_P2P_IOCTL_UNMAP, &request);
      }
      peerMappings_.clear();
      for (const auto &[iova, bytes] : dmaMappings_) {
        unmapIoas(iova, bytes);
      }
      dmaMappings_.clear();
      releaseDma(queueMemory_);
      releaseDma(adminMemory_);
    }
    peerMapper_.reset();
    if (doorbellBar_ != nullptr) {
      (void)::munmap(doorbellBar_, pageSize_);
      doorbellBar_ = nullptr;
    }
    if (bar_ != nullptr) {
      (void)::munmap(bar_, pageSize_);
      bar_ = nullptr;
    }
    if (attached_ && vfioDevice_.get() >= 0) {
      vfio_device_detach_iommufd_pt detach{};
      detach.argsz = sizeof(detach);
      (void)::ioctl(vfioDevice_.get(), VFIO_DEVICE_DETACH_IOMMUFD_PT, &detach);
      attached_ = false;
    }
    vfioDevice_.reset();
    if (iommufd_.get() >= 0 && ioasId_ != 0) {
      iommu_destroy destroy{sizeof(destroy), ioasId_};
      (void)::ioctl(iommufd_.get(), IOMMU_DESTROY, &destroy);
      ioasId_ = 0;
    }
    iommufd_.reset();
  }

  void setBusMaster(bool enabled) {
    std::uint16_t command = 0;
    const off_t offset =
        static_cast<off_t>(configRegion_.offset + PciCommandOffset);
    if (::pread(vfioDevice_.get(), &command, sizeof(command), offset) !=
        static_cast<ssize_t>(sizeof(command))) {
      throwSystem("pread VFIO PCI command register");
    }
    if constexpr (std::endian::native == std::endian::big) {
      command = byteSwap(command);
    }
    command = enabled ? static_cast<std::uint16_t>(command | PciCommandMemory |
                                                   PciCommandMaster)
                      : static_cast<std::uint16_t>(command & ~PciCommandMaster);
    if constexpr (std::endian::native == std::endian::big) {
      command = byteSwap(command);
    }
    if (::pwrite(vfioDevice_.get(), &command, sizeof(command), offset) !=
        static_cast<ssize_t>(sizeof(command))) {
      throwSystem("pwrite VFIO PCI command register");
    }
  }

  void setBusMasterNoexcept(bool enabled) noexcept {
    try {
      if (vfioDevice_.get() >= 0 && configRegion_.size != 0) {
        setBusMaster(enabled);
      }
    } catch (...) {
    }
  }

  std::uint32_t readMmio32(std::size_t offset) const noexcept {
    const std::uint32_t value = *reinterpret_cast<volatile std::uint32_t *>(
        static_cast<std::byte *>(bar_) + offset);
    return fromLittle32(value);
  }

  std::uint64_t readMmio64(std::size_t offset) const noexcept {
    const std::uint64_t value = *reinterpret_cast<volatile std::uint64_t *>(
        static_cast<std::byte *>(bar_) + offset);
    return fromLittle64(value);
  }

  void writeMmio32(std::size_t offset, std::uint32_t value) noexcept {
    *reinterpret_cast<volatile std::uint32_t *>(static_cast<std::byte *>(bar_) +
                                                offset) = toLittle32(value);
  }

  void writeMmio64(std::size_t offset, std::uint64_t value) noexcept {
    *reinterpret_cast<volatile std::uint64_t *>(static_cast<std::byte *>(bar_) +
                                                offset) = toLittle64(value);
  }

  void writeDoorbell(std::uint32_t qid, bool completion,
                     std::uint32_t value) noexcept {
    const std::size_t index = 2U * qid + (completion ? 1U : 0U);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile std::uint32_t *>(
        static_cast<std::byte *>(doorbellBar_) + index * doorbellStride_) =
        toLittle32(value);
  }

  void waitReady(bool ready) {
    const auto deadline = std::chrono::steady_clock::now() + controllerTimeout_;
    while (true) {
      const std::uint32_t status = readMmio32(NvmeRegisterControllerStatus);
      if (((status & NvmeControllerReady) != 0) == ready) {
        return;
      }
      if ((status & NvmeControllerFatal) != 0 ||
          std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("NVMe controller ready transition failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  void waitReadyNoexcept(bool ready) noexcept {
    try {
      waitReady(ready);
    } catch (...) {
      fatal_ = true;
    }
  }

  std::string bdf_;
  std::uint32_t namespaceId_ = 0;
  std::uint32_t requestedQueueDepth_ = 0;
  std::chrono::milliseconds adminTimeout_{};
  NvmeMediaPolicy mediaPolicy_ =
      NvmeMediaPolicy::RequireHardwareWriteProtection;
  std::chrono::milliseconds controllerTimeout_{500};
  std::size_t pageSize_ = 0;
  FileDescriptor iommufd_;
  FileDescriptor vfioDevice_;
  FileDescriptor peerMapper_;
  std::uint32_t ioasId_ = 0;
  bool attached_ = false;
  vfio_region_info barRegion_{};
  vfio_region_info configRegion_{};
  void *bar_ = nullptr;
  void *doorbellBar_ = nullptr;
  std::uint64_t cap_ = 0;
  std::uint32_t doorbellStride_ = 0;
  DmaRegion adminMemory_{};
  DmaRegion queueMemory_{};
  std::size_t adminSqOffset_ = 0;
  std::size_t adminCqOffset_ = 0;
  std::size_t identifyOffset_ = 0;
  std::size_t controlOffset_ = 0;
  std::size_t sqOffset_ = 0;
  std::size_t cqOffset_ = 0;
  std::size_t prpOffset_ = 0;
  Submission *adminSq_ = nullptr;
  Completion *adminCq_ = nullptr;
  std::byte *identify_ = nullptr;
  std::uint32_t adminTail_ = 0;
  std::uint32_t adminHead_ = 0;
  std::uint16_t adminCid_ = 1;
  std::uint16_t adminPhase_ = 1;
  std::uint32_t namespaceCount_ = 0;
  std::uint64_t namespaceBlocks_ = 0;
  std::uint32_t maxTransferBytes_ = 0;
  std::uint32_t queueCount_ = 0;
  std::uint32_t queueId_ = 1;
  std::uint32_t generation_ = 1;
  std::uint8_t lbaShift_ = 0;
  bool namespaceReadOnly_ = false;
  bool writeProtectSupported_ = false;
  bool controllerEnabled_ = false;
  bool resetPerformed_ = false;
  bool sqLive_ = false;
  bool cqLive_ = false;
  bool quiesced_ = false;
  bool fatal_ = false;
  NvmeQueueResources resources_{};
  std::unordered_map<std::uint64_t, std::size_t> dmaMappings_;
  std::unordered_set<std::uint64_t> peerMappings_;
  std::mutex mutex_;
};

} // namespace

std::unique_ptr<NvmeControlPlane>
createVfioNvmeControlPlane(const NvmeTransportOptions &options) {
  return std::make_unique<VfioNvmeControlPlane>(options);
}

} // namespace nta::detail
