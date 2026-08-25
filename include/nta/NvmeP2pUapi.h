#pragma once

#include <linux/ioctl.h>
#include <linux/types.h>

#define NTA_NVME_P2P_ABI_VERSION 1U
#define NTA_NVME_P2P_DEVICE_PATH "/dev/nta_nvme_p2p"

/*
 * Map a pinned GPU virtual-address range for DMA by one VFIO-owned NVMe PCI
 * function. The range must be 64 KiB aligned, as required by NVIDIA's
 * persistent peer-memory API. dma_addresses points to a userspace array with
 * dma_capacity entries. On success, page_size and entry_count describe the
 * native DMA-page vector written to that array and handle owns the mapping.
 */
struct nta_nvme_p2p_map {
  __u32 size;
  __u32 abi_version;
  __u64 gpu_address;
  __u64 bytes;
  __u32 pci_domain;
  __u32 pci_bus;
  __u32 pci_device;
  __u32 pci_function;
  __u32 page_size;
  __u32 entry_count;
  __u64 dma_addresses;
  __u32 dma_capacity;
  __u32 reserved;
  __u64 handle;
};

struct nta_nvme_p2p_unmap {
  __u32 size;
  __u32 abi_version;
  __u64 handle;
};

#define NTA_NVME_P2P_IOCTL_MAGIC 'N'
#define NTA_NVME_P2P_IOCTL_MAP                                              \
  _IOWR(NTA_NVME_P2P_IOCTL_MAGIC, 1, struct nta_nvme_p2p_map)
#define NTA_NVME_P2P_IOCTL_UNMAP                                            \
  _IOW(NTA_NVME_P2P_IOCTL_MAGIC, 2, struct nta_nvme_p2p_unmap)
