#pragma once

#include <linux/ioctl.h>
#include <linux/types.h>

#define NTA_NVME_ABI_VERSION 2U
#define NTA_NVME_IOCTL_MAGIC 0xB7

#define NTA_NVME_MMAP_QUEUE_PGOFF 0x100ULL
#define NTA_NVME_MMAP_DOORBELL_PGOFF 0x300ULL

#define NTA_NVME_IMPORT_REQUIRE_CONTIGUOUS (1U << 0)
#define NTA_NVME_MAX_DMA_PAGES 256U

#define NTA_NVME_CAP_IOMMU_TRANSLATED (1U << 0)
#define NTA_NVME_CAP_NAMESPACE_READ_ONLY (1U << 1)
#define NTA_NVME_CAP_STATIC_DMA_BUF (1U << 2)
#define NTA_NVME_CAP_MULTI_QUEUE (1U << 3)
#define NTA_NVME_CAP_TRUSTED_RAW_QUEUE (1U << 4)

#define NTA_NVME_QUEUE_CONTROL_MAGIC 0x4e544151U

enum nta_nvme_queue_state {
  NTA_NVME_QUEUE_OFFLINE = 0,
  NTA_NVME_QUEUE_ONLINE = 1,
  NTA_NVME_QUEUE_QUIESCED = 2,
  NTA_NVME_QUEUE_FATAL = 3,
  NTA_NVME_QUEUE_REMOVED = 4,
};

/* The first controller page of every queue mapping is driver-populated. */
struct nta_nvme_queue_control {
  __u32 magic;
  __u32 abi_version;
  __u32 state;
  __u32 generation;
  __u32 queue_id;
  __u32 reserved0;
  __u64 reserved1[5];
};

struct nta_nvme_info {
  __u32 abi_version;
  __u32 queue_depth;
  __u32 controller_page_size;
  __u32 lba_shift;
  __u32 namespace_id;
  __u32 doorbell_stride;
  __u32 max_transfer_bytes;
  __u32 capabilities;
  __u32 queue_id;
  __u32 queue_count;
  __u32 generation;
  __u32 reserved0;
  __u64 namespace_blocks;
  __u64 queue_bytes;
  __u64 control_offset;
  __u64 sq_offset;
  __u64 cq_offset;
  __u64 prp_offset;
  __u64 prp_dma_address;
  __u64 sq_doorbell_offset;
  __u64 cq_doorbell_offset;
  __u64 doorbell_mmap_bytes;
};

struct nta_nvme_import {
  __s32 dma_buf_fd;
  __u32 flags;
  __u64 bytes;
  __u64 handle;
  __u64 dma_address;
  __u64 mapped_bytes;
  __u32 dma_segments;
  __u32 reserved0;
};

struct nta_nvme_release {
  __u64 handle;
};

struct nta_nvme_register_host {
  __u64 user_address;
  __u64 bytes;
  __u64 handle;
  __u32 dma_pages;
  __u32 reserved0;
};

struct nta_nvme_dma_pages {
  __u64 handle;
  __u32 first_page;
  __u32 page_count;
  __u64 addresses[NTA_NVME_MAX_DMA_PAGES];
};

#define NTA_NVME_IOCTL_GET_INFO                                                \
  _IOR(NTA_NVME_IOCTL_MAGIC, 0x00, struct nta_nvme_info)
#define NTA_NVME_IOCTL_IMPORT_DMA_BUF                                          \
  _IOWR(NTA_NVME_IOCTL_MAGIC, 0x01, struct nta_nvme_import)
#define NTA_NVME_IOCTL_RELEASE_DMA_BUF                                         \
  _IOW(NTA_NVME_IOCTL_MAGIC, 0x02, struct nta_nvme_release)
#define NTA_NVME_IOCTL_REGISTER_HOST                                           \
  _IOWR(NTA_NVME_IOCTL_MAGIC, 0x03, struct nta_nvme_register_host)
#define NTA_NVME_IOCTL_GET_DMA_PAGES                                           \
  _IOWR(NTA_NVME_IOCTL_MAGIC, 0x04, struct nta_nvme_dma_pages)
#define NTA_NVME_IOCTL_QUIESCE _IO(NTA_NVME_IOCTL_MAGIC, 0x05)
