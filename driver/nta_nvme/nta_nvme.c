// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/capability.h>
#include <linux/delay.h>
#include <linux/dma-buf.h>
#include <linux/dma-fence.h>
#include <linux/dma-resv.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/iommu.h>
#include <linux/iopoll.h>
#include <linux/kref.h>
#include <linux/list.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/nvme.h>
#include <linux/pci.h>
#include <linux/pm.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/xarray.h>

#include "nta/NvmeUapi.h"

#define NTA_DRIVER_NAME "nta_nvme"
#define NTA_ADMIN_DEPTH 32U
#define NTA_IO_DEPTH 64U
#define NTA_ADMIN_TIMEOUT_US (10U * 1000U * 1000U)
#define NTA_READY_POLL_US 1000U
#define NTA_MAX_IO_QUEUES 32U

static char target_bdf[16] = "0000:d8:00.0";
module_param_string(target_bdf, target_bdf, sizeof(target_bdf), 0444);
MODULE_PARM_DESC(target_bdf,
                 "Only bind the exact PCI domain:bus:slot.function");

static uint target_nsid = 1;
module_param(target_nsid, uint, 0444);
MODULE_PARM_DESC(target_nsid, "Read-only namespace exposed to GPU queues");

static uint max_io_queues = 8;
module_param(max_io_queues, uint, 0444);
MODULE_PARM_DESC(max_io_queues, "Maximum per-open GPU I/O queues");

struct nta_nvme_dev;

struct nta_mapping_fence {
  struct dma_fence base;
  spinlock_t lock;
};

struct nta_dma_mapping {
  struct dma_buf *dmabuf;
  struct dma_buf_attachment *attachment;
  struct sg_table *sgt;
  struct nta_mapping_fence *fence;
  struct page **pages;
  unsigned long page_count;
  struct sg_table host_sgt;
  bool host_mapping;
  bool host_sgt_valid;
  bool host_dma_mapped;
  u64 handle;
};

struct nta_nvme_queue {
  struct nta_nvme_dev *dev;
  struct list_head node;
  struct kref refs;
  struct xarray mappings;
  u64 next_mapping;
  u16 qid;
  bool live;
  bool detached;

  void *memory;
  dma_addr_t memory_dma;
  size_t bytes;
  size_t control_offset;
  size_t sq_offset;
  size_t cq_offset;
  size_t prp_offset;
  struct nta_nvme_queue_control *control;
};

struct nta_nvme_dev {
  struct pci_dev *pdev;
  void __iomem *bar;
  resource_size_t bar_start;
  u64 cap;
  u32 doorbell_stride;
  u32 controller_page_size;
  u32 lba_shift;
  u32 max_transfer_bytes;
  u32 queue_count;
  u32 generation;
  u32 namespace_count;
  u64 namespace_blocks;
  bool namespace_write_protected;
  bool fatal;
  bool removing;
  bool suspended;

  struct nvme_command *admin_sq;
  dma_addr_t admin_sq_dma;
  struct nvme_completion *admin_cq;
  dma_addr_t admin_cq_dma;
  u16 admin_tail;
  u16 admin_head;
  u16 admin_cid;
  u8 admin_phase;

  struct miscdevice misc;
  struct mutex lock;
  struct list_head queues;
  unsigned long *queue_bitmap;
  u32 open_count;
  atomic_t queue_refs;
  wait_queue_head_t close_wait;
};

static const char *nta_fence_driver_name(struct dma_fence *fence) {
  return NTA_DRIVER_NAME;
}

static const char *nta_fence_timeline_name(struct dma_fence *fence) {
  return "queue-mapping";
}

static void nta_fence_release(struct dma_fence *fence) {
  struct nta_mapping_fence *mapping_fence =
      container_of(fence, struct nta_mapping_fence, base);

  kfree(mapping_fence);
}

static const struct dma_fence_ops nta_fence_ops = {
    .get_driver_name = nta_fence_driver_name,
    .get_timeline_name = nta_fence_timeline_name,
    .release = nta_fence_release,
};

static struct nta_mapping_fence *nta_fence_create(void) {
  struct nta_mapping_fence *fence = kzalloc(sizeof(*fence), GFP_KERNEL);

  if (!fence)
    return NULL;
  spin_lock_init(&fence->lock);
  dma_fence_init(&fence->base, &nta_fence_ops, &fence->lock,
                 dma_fence_context_alloc(1), 1);
  return fence;
}

static void __iomem *nta_doorbell(struct nta_nvme_dev *dev, u16 qid,
                                  bool completion) {
  u32 index = 2U * qid + (completion ? 1U : 0U);

  return dev->bar + NVME_REG_DBS + index * dev->doorbell_stride;
}

static void nta_set_queue_state(struct nta_nvme_queue *queue, u32 state) {
  WRITE_ONCE(queue->control->generation, queue->dev->generation);
  dma_wmb();
  WRITE_ONCE(queue->control->state, state);
}

static void nta_mark_fatal_locked(struct nta_nvme_dev *dev, u32 state) {
  struct nta_nvme_queue *queue;

  lockdep_assert_held(&dev->lock);
  if (!dev->fatal)
    ++dev->generation;
  dev->fatal = true;
  pci_clear_master(dev->pdev);
  list_for_each_entry(queue, &dev->queues, node)
    nta_set_queue_state(queue, state);
}

static int nta_wait_ready(struct nta_nvme_dev *dev, bool ready) {
  u32 csts;
  u32 timeout_ms = max_t(u32, 500U, NVME_CAP_TIMEOUT(dev->cap) * 500U);

  return readl_poll_timeout(dev->bar + NVME_REG_CSTS, csts,
                            !!(csts & NVME_CSTS_RDY) == ready,
                            NTA_READY_POLL_US, timeout_ms * 1000U);
}

static int nta_disable_controller(struct nta_nvme_dev *dev) {
  u32 cc = readl(dev->bar + NVME_REG_CC);

  if (!(cc & NVME_CC_ENABLE))
    return 0;
  writel(cc & ~NVME_CC_ENABLE, dev->bar + NVME_REG_CC);
  return nta_wait_ready(dev, false);
}

static int nta_enable_controller(struct nta_nvme_dev *dev) {
  u32 cc;
  u32 mps = ilog2(dev->controller_page_size) - 12U;

  memset(dev->admin_sq, 0, NTA_ADMIN_DEPTH * sizeof(*dev->admin_sq));
  memset(dev->admin_cq, 0, NTA_ADMIN_DEPTH * sizeof(*dev->admin_cq));
  dev->admin_tail = 0;
  dev->admin_head = 0;
  dev->admin_cid = 1;
  dev->admin_phase = 1;

  writel(~0U, dev->bar + NVME_REG_INTMS);
  writel(((NTA_ADMIN_DEPTH - 1U) << 16) | (NTA_ADMIN_DEPTH - 1U),
         dev->bar + NVME_REG_AQA);
  writeq(dev->admin_sq_dma, dev->bar + NVME_REG_ASQ);
  writeq(dev->admin_cq_dma, dev->bar + NVME_REG_ACQ);

  cc = NVME_CC_CSS_NVM | (mps << NVME_CC_MPS_SHIFT) | NVME_CC_AMS_RR |
       NVME_CC_IOSQES | NVME_CC_IOCQES | NVME_CC_ENABLE;
  writel(cc, dev->bar + NVME_REG_CC);
  return nta_wait_ready(dev, true);
}

static int nta_admin_command(struct nta_nvme_dev *dev,
                             struct nvme_command *command, u32 *result) {
  struct nvme_completion *completion;
  unsigned long timeout;
  u16 status;
  u16 cid = dev->admin_cid++;

  lockdep_assert_held(&dev->lock);
  if (dev->fatal || dev->removing)
    return -EIO;
  command->common.command_id = cpu_to_le16(cid);
  memcpy(&dev->admin_sq[dev->admin_tail], command, sizeof(*command));
  dev->admin_tail = (dev->admin_tail + 1U) % NTA_ADMIN_DEPTH;
  dma_wmb();
  writel(dev->admin_tail, nta_doorbell(dev, 0, false));

  timeout = jiffies + usecs_to_jiffies(NTA_ADMIN_TIMEOUT_US);
  completion = &dev->admin_cq[dev->admin_head];
  do {
    status = le16_to_cpu(READ_ONCE(completion->status));
    if ((status & 1U) == dev->admin_phase)
      break;
    if (time_after(jiffies, timeout)) {
      dev_err(&dev->pdev->dev, "admin opcode %#x timed out; queue poisoned\n",
              command->common.opcode);
      nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
      return -ETIMEDOUT;
    }
    usleep_range(50, 100);
  } while (true);
  dma_rmb();

  if (le16_to_cpu(completion->command_id) != cid) {
    nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
    return -EIO;
  }
  if (result)
    *result = le32_to_cpu(completion->result.u32);

  dev->admin_head++;
  if (dev->admin_head == NTA_ADMIN_DEPTH) {
    dev->admin_head = 0;
    dev->admin_phase ^= 1U;
  }
  writel(dev->admin_head, nta_doorbell(dev, 0, true));

  status >>= 1;
  if (status) {
    dev_err(&dev->pdev->dev, "admin opcode %#x failed with status %#x\n",
            command->common.opcode, status);
    return -EIO;
  }
  return 0;
}

static int nta_identify(struct nta_nvme_dev *dev) {
  struct nvme_id_ctrl *ctrl;
  struct nvme_id_ns *ns;
  struct nvme_command command = {};
  dma_addr_t identify_dma;
  void *identify;
  u8 format;
  int ret;

  identify = dma_alloc_coherent(&dev->pdev->dev, NVME_IDENTIFY_DATA_SIZE,
                                &identify_dma, GFP_KERNEL);
  if (!identify)
    return -ENOMEM;

  command.identify.opcode = nvme_admin_identify;
  command.identify.dptr.prp1 = cpu_to_le64(identify_dma);
  command.identify.cns = NVME_ID_CNS_CTRL;
  ret = nta_admin_command(dev, &command, NULL);
  if (ret)
    goto out;

  ctrl = identify;
  if (target_nsid == 0 || target_nsid > le32_to_cpu(ctrl->nn)) {
    dev_err(&dev->pdev->dev, "namespace %u is not present\n", target_nsid);
    ret = -ENODEV;
    goto out;
  }
  dev->namespace_count = le32_to_cpu(ctrl->nn);
  if (!(ctrl->nwpc & 1U)) {
    dev_err(&dev->pdev->dev,
            "controller lacks basic namespace write protection\n");
    ret = -EOPNOTSUPP;
    goto out;
  }
  if (ctrl->mdts && ctrl->mdts < 9U)
    dev->max_transfer_bytes = dev->controller_page_size << ctrl->mdts;
  else
    dev->max_transfer_bytes = dev->controller_page_size * 512U;

  memset(identify, 0, NVME_IDENTIFY_DATA_SIZE);
  memset(&command, 0, sizeof(command));
  command.identify.opcode = nvme_admin_identify;
  command.identify.nsid = cpu_to_le32(target_nsid);
  command.identify.dptr.prp1 = cpu_to_le64(identify_dma);
  command.identify.cns = NVME_ID_CNS_NS;
  ret = nta_admin_command(dev, &command, NULL);
  if (ret)
    goto out;

  ns = identify;
  format = nvme_lbaf_index(ns->flbas);
  if (le64_to_cpu(ns->nsze) == 0 || format > ns->nlbaf ||
      ns->lbaf[format].ds < 9U || ns->lbaf[format].ds > 16U ||
      le16_to_cpu(ns->lbaf[format].ms) != 0U ||
      (ns->dps & NVME_NS_DPS_PI_MASK) != 0U) {
    dev_err(&dev->pdev->dev,
            "namespace %u requires unsupported metadata or protection\n",
            target_nsid);
    ret = -EOPNOTSUPP;
    goto out;
  }
  dev->lba_shift = ns->lbaf[format].ds;
  dev->namespace_blocks = le64_to_cpu(ns->nsze);

out:
  dma_free_coherent(&dev->pdev->dev, NVME_IDENTIFY_DATA_SIZE, identify,
                    identify_dma);
  return ret;
}

static int nta_set_namespace_read_only(struct nta_nvme_dev *dev, u32 nsid) {
  struct nvme_command command = {};
  u32 result = 0;
  int ret;

  command.features.opcode = nvme_admin_set_features;
  command.features.nsid = cpu_to_le32(nsid);
  command.features.fid = cpu_to_le32(NVME_FEAT_WRITE_PROTECT);
  command.features.dword11 = cpu_to_le32(NVME_NS_WRITE_PROTECT);
  ret = nta_admin_command(dev, &command, NULL);
  if (ret)
    return ret;

  memset(&command, 0, sizeof(command));
  command.features.opcode = nvme_admin_get_features;
  command.features.nsid = cpu_to_le32(nsid);
  command.features.fid = cpu_to_le32(NVME_FEAT_WRITE_PROTECT);
  ret = nta_admin_command(dev, &command, &result);
  if (ret)
    return ret;
  if ((result & 0x7U) != NVME_NS_WRITE_PROTECT) {
    dev_err(&dev->pdev->dev,
            "namespace %u did not enter basic write-protected state\n",
            nsid);
    return -EACCES;
  }
  return 0;
}

static int nta_protect_active_namespaces(struct nta_nvme_dev *dev) {
  struct nvme_command command = {};
  dma_addr_t list_dma;
  __le32 *list;
  u32 cursor = 0;
  u32 protected = 0;
  bool target_found = false;
  int ret = 0;

  list = dma_alloc_coherent(&dev->pdev->dev, NVME_IDENTIFY_DATA_SIZE,
                            &list_dma, GFP_KERNEL);
  if (!list)
    return -ENOMEM;
  do {
    u32 index;

    memset(list, 0, NVME_IDENTIFY_DATA_SIZE);
    memset(&command, 0, sizeof(command));
    command.identify.opcode = nvme_admin_identify;
    command.identify.nsid = cpu_to_le32(cursor);
    command.identify.dptr.prp1 = cpu_to_le64(list_dma);
    command.identify.cns = NVME_ID_CNS_NS_ACTIVE_LIST;
    ret = nta_admin_command(dev, &command, NULL);
    if (ret)
      break;

    for (index = 0; index < NVME_IDENTIFY_DATA_SIZE / sizeof(*list); ++index) {
      u32 nsid = le32_to_cpu(list[index]);

      if (!nsid)
        break;
      if (nsid <= cursor) {
        ret = -EIO;
        goto out;
      }
      ret = nta_set_namespace_read_only(dev, nsid);
      if (ret)
        goto out;
      target_found |= nsid == target_nsid;
      cursor = nsid;
      ++protected;
    }
    if (!index || index < NVME_IDENTIFY_DATA_SIZE / sizeof(*list))
      break;
  } while (protected < dev->namespace_count);

  if (!ret && !target_found)
    ret = -ENODEV;
  if (!ret)
    dev->namespace_write_protected = true;

out:
  dma_free_coherent(&dev->pdev->dev, NVME_IDENTIFY_DATA_SIZE, list, list_dma);
  return ret;
}

static int nta_configure_io_queues(struct nta_nvme_dev *dev) {
  struct nvme_command command = {};
  unsigned long *bitmap;
  u32 requested = clamp_t(u32, max_io_queues, 1U, NTA_MAX_IO_QUEUES);
  u32 result;
  u32 allocated;
  int ret;

  command.features.opcode = nvme_admin_set_features;
  command.features.fid = cpu_to_le32(NVME_FEAT_NUM_QUEUES);
  command.features.dword11 =
      cpu_to_le32((requested - 1U) | ((requested - 1U) << 16U));
  ret = nta_admin_command(dev, &command, &result);
  if (ret)
    return ret;
  allocated = min(result & 0xffffU, result >> 16U) + 1U;
  allocated = min(allocated, requested);
  if (!allocated)
    return -ENOSPC;
  if ((2U * allocated + 1U) * dev->doorbell_stride + sizeof(u32) > PAGE_SIZE) {
    dev_err(&dev->pdev->dev, "queue doorbells exceed the isolated BAR page\n");
    return -EOPNOTSUPP;
  }

  bitmap = bitmap_zalloc(allocated + 1U, GFP_KERNEL);
  if (!bitmap)
    return -ENOMEM;
  __set_bit(0, bitmap);
  bitmap_free(dev->queue_bitmap);
  dev->queue_bitmap = bitmap;
  dev->queue_count = allocated;
  return 0;
}

static int nta_start_controller_locked(struct nta_nvme_dev *dev) {
  int ret;

  lockdep_assert_held(&dev->lock);
  dev->fatal = false;
  dev->namespace_write_protected = false;
  pci_set_master(dev->pdev);
  ret = nta_disable_controller(dev);
  if (ret)
    goto fatal;
  ret = nta_enable_controller(dev);
  if (ret)
    goto fatal;
  ret = nta_identify(dev);
  if (ret)
    goto fatal;
  ret = nta_protect_active_namespaces(dev);
  if (ret)
    goto fatal;
  ret = nta_configure_io_queues(dev);
  if (ret)
    goto fatal;
  return 0;

fatal:
  nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
  return ret;
}

static int nta_recover_controller_locked(struct nta_nvme_dev *dev) {
  int ret;

  lockdep_assert_held(&dev->lock);
  if (dev->open_count || atomic_read(&dev->queue_refs))
    return -EBUSY;
  pci_clear_master(dev->pdev);
  ret = pci_reset_function(dev->pdev);
  if (ret)
    return ret;
  ++dev->generation;
  return nta_start_controller_locked(dev);
}

static int nta_create_io_queue_locked(struct nta_nvme_queue *queue) {
  struct nta_nvme_dev *dev = queue->dev;
  struct nvme_command command = {};
  int ret;

  command.create_cq.opcode = nvme_admin_create_cq;
  command.create_cq.prp1 = cpu_to_le64(queue->memory_dma + queue->cq_offset);
  command.create_cq.cqid = cpu_to_le16(queue->qid);
  command.create_cq.qsize = cpu_to_le16(NTA_IO_DEPTH - 1U);
  command.create_cq.cq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
  ret = nta_admin_command(dev, &command, NULL);
  if (ret)
    return ret;

  memset(&command, 0, sizeof(command));
  command.create_sq.opcode = nvme_admin_create_sq;
  command.create_sq.prp1 = cpu_to_le64(queue->memory_dma + queue->sq_offset);
  command.create_sq.sqid = cpu_to_le16(queue->qid);
  command.create_sq.qsize = cpu_to_le16(NTA_IO_DEPTH - 1U);
  command.create_sq.sq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
  command.create_sq.cqid = cpu_to_le16(queue->qid);
  ret = nta_admin_command(dev, &command, NULL);
  if (ret) {
    if (!dev->fatal) {
      int cleanup_ret;

      memset(&command, 0, sizeof(command));
      command.delete_queue.opcode = nvme_admin_delete_cq;
      command.delete_queue.qid = cpu_to_le16(queue->qid);
      cleanup_ret = nta_admin_command(dev, &command, NULL);
      if (cleanup_ret)
        nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
    }
    return ret;
  }
  queue->live = true;
  nta_set_queue_state(queue, NTA_NVME_QUEUE_ONLINE);
  return 0;
}

static int nta_delete_io_queue_locked(struct nta_nvme_queue *queue) {
  struct nta_nvme_dev *dev = queue->dev;
  struct nvme_command command = {};
  int ret = 0;

  if (!queue->live)
    return 0;
  nta_set_queue_state(queue, NTA_NVME_QUEUE_QUIESCED);
  if (dev->fatal || dev->removing) {
    queue->live = false;
    return -EIO;
  }

  command.delete_queue.opcode = nvme_admin_delete_sq;
  command.delete_queue.qid = cpu_to_le16(queue->qid);
  ret = nta_admin_command(dev, &command, NULL);
  if (!ret) {
    memset(&command, 0, sizeof(command));
    command.delete_queue.opcode = nvme_admin_delete_cq;
    command.delete_queue.qid = cpu_to_le16(queue->qid);
    ret = nta_admin_command(dev, &command, NULL);
  }
  if (ret)
    nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
  queue->live = false;
  return ret;
}

static void nta_release_mapping(struct nta_nvme_queue *queue,
                                struct nta_dma_mapping *mapping) {
  struct nta_nvme_dev *dev = queue->dev;

  if (!mapping)
    return;
  if (mapping->fence) {
    dma_fence_signal(&mapping->fence->base);
    dma_fence_put(&mapping->fence->base);
  }
  if (mapping->host_mapping && mapping->host_dma_mapped)
    dma_unmap_sgtable(&dev->pdev->dev, &mapping->host_sgt, DMA_BIDIRECTIONAL,
                      0);
  else if (mapping->sgt)
    dma_buf_unmap_attachment_unlocked(mapping->attachment, mapping->sgt,
                                      DMA_BIDIRECTIONAL);
  if (mapping->attachment)
    dma_buf_detach(mapping->dmabuf, mapping->attachment);
  if (mapping->dmabuf)
    dma_buf_put(mapping->dmabuf);
  if (mapping->host_sgt_valid)
    sg_free_table(&mapping->host_sgt);
  if (mapping->pages) {
    unpin_user_pages_dirty_lock(mapping->pages, mapping->page_count, true);
    kvfree(mapping->pages);
  }
  kfree(mapping);
}

static void nta_release_all_mappings(struct nta_nvme_queue *queue) {
  struct nta_dma_mapping *mapping;
  unsigned long index;

  xa_for_each(&queue->mappings, index, mapping) {
    xa_erase(&queue->mappings, index);
    nta_release_mapping(queue, mapping);
  }
}

static void nta_queue_free(struct kref *refs) {
  struct nta_nvme_queue *queue =
      container_of(refs, struct nta_nvme_queue, refs);
  struct nta_nvme_dev *dev = queue->dev;

  xa_destroy(&queue->mappings);
  dma_free_coherent(&dev->pdev->dev, queue->bytes, queue->memory,
                    queue->memory_dma);
  kfree(queue);
  atomic_dec(&dev->queue_refs);
  wake_up_all(&dev->close_wait);
}

static u32 nta_dma_page_count(const struct nta_dma_mapping *mapping,
                              u32 page_size) {
  struct scatterlist *sg;
  u64 pages = 0;
  unsigned int index;

  for_each_sgtable_dma_sg(mapping->sgt, sg, index)
    pages += DIV_ROUND_UP(sg_dma_len(sg), page_size);
  return min_t(u64, pages, U32_MAX);
}

static int nta_validate_prp_mapping(const struct nta_dma_mapping *mapping,
                                    u32 page_size, u64 bytes) {
  struct scatterlist *sg;
  u64 covered = 0;
  unsigned int index;

  for_each_sgtable_dma_sg(mapping->sgt, sg, index) {
    u64 length = sg_dma_len(sg);

    if (!IS_ALIGNED(sg_dma_address(sg), page_size) ||
        length > U64_MAX - covered)
      return -ERANGE;
    covered += length;
    if (covered < bytes && !IS_ALIGNED(length, page_size))
      return -ERANGE;
    if (covered >= bytes)
      return 0;
  }
  return -ERANGE;
}

static int nta_dma_pages(struct nta_nvme_queue *queue,
                         struct nta_nvme_dma_pages *request) {
  struct nta_dma_mapping *mapping = xa_load(&queue->mappings, request->handle);
  struct scatterlist *sg;
  u32 logical_page = 0;
  u32 copied = 0;
  unsigned int index;

  if (!mapping)
    return -ENOENT;
  if (!request->page_count || request->page_count > NTA_NVME_MAX_DMA_PAGES)
    return -EINVAL;

  for_each_sgtable_dma_sg(mapping->sgt, sg, index) {
    dma_addr_t address = sg_dma_address(sg);
    u32 pages = DIV_ROUND_UP(sg_dma_len(sg), queue->dev->controller_page_size);
    u32 page;

    for (page = 0; page < pages; ++page, ++logical_page) {
      if (logical_page < request->first_page)
        continue;
      if (copied == request->page_count)
        goto done;
      request->addresses[copied++] =
          address + (u64)page * queue->dev->controller_page_size;
    }
  }
done:
  request->page_count = copied;
  return copied ? 0 : -ERANGE;
}

static int nta_register_host(struct nta_nvme_queue *queue,
                             struct nta_nvme_register_host *request) {
  struct nta_nvme_dev *dev = queue->dev;
  struct nta_dma_mapping *mapping;
  unsigned long address = request->user_address;
  unsigned long pages;
  long pinned;
  int ret;

  if (!queue->live)
    return -ESHUTDOWN;
  if (!request->bytes || !PAGE_ALIGNED(address) ||
      !PAGE_ALIGNED(request->bytes))
    return -EINVAL;
  pages = request->bytes >> PAGE_SHIFT;
  if (pages > U32_MAX)
    return -E2BIG;

  mapping = kzalloc(sizeof(*mapping), GFP_KERNEL);
  if (!mapping)
    return -ENOMEM;
  mapping->host_mapping = true;
  mapping->page_count = pages;
  mapping->pages = kvmalloc_array(pages, sizeof(*mapping->pages), GFP_KERNEL);
  if (!mapping->pages) {
    ret = -ENOMEM;
    goto fail;
  }
  pinned = pin_user_pages_fast(address, pages, FOLL_WRITE | FOLL_LONGTERM,
                               mapping->pages);
  if (pinned != pages) {
    if (pinned > 0) {
      mapping->page_count = pinned;
      unpin_user_pages(mapping->pages, mapping->page_count);
    }
    mapping->page_count = 0;
    ret = pinned < 0 ? pinned : -EFAULT;
    goto fail;
  }
  ret = sg_alloc_table_from_pages(&mapping->host_sgt, mapping->pages, pages, 0,
                                  request->bytes, GFP_KERNEL);
  if (ret)
    goto fail;
  mapping->host_sgt_valid = true;
  ret = dma_map_sgtable(&dev->pdev->dev, &mapping->host_sgt,
                        DMA_BIDIRECTIONAL, 0);
  if (ret)
    goto fail;
  mapping->host_dma_mapped = true;
  mapping->sgt = &mapping->host_sgt;
  ret = nta_validate_prp_mapping(mapping, dev->controller_page_size,
                                 request->bytes);
  if (ret)
    goto fail;
  mapping->handle = ++queue->next_mapping;
  ret = xa_insert(&queue->mappings, mapping->handle, mapping, GFP_KERNEL);
  if (ret)
    goto fail;

  request->handle = mapping->handle;
  request->dma_pages = nta_dma_page_count(mapping, dev->controller_page_size);
  return 0;

fail:
  nta_release_mapping(queue, mapping);
  return ret;
}

static int nta_add_mapping_fence(struct nta_dma_mapping *mapping) {
  struct dma_resv *resv = mapping->dmabuf->resv;
  int ret;

  mapping->fence = nta_fence_create();
  if (!mapping->fence)
    return -ENOMEM;
  ret = dma_resv_lock(resv, NULL);
  if (ret)
    return ret;
  ret = dma_resv_reserve_fences(resv, 1);
  if (!ret)
    dma_resv_add_fence(resv, &mapping->fence->base, DMA_RESV_USAGE_WRITE);
  dma_resv_unlock(resv);
  return ret;
}

static int nta_import_dmabuf(struct nta_nvme_queue *queue,
                             struct nta_nvme_import *request) {
  struct nta_dma_mapping *mapping;
  struct scatterlist *sg;
  u64 total = 0;
  unsigned int index;
  int ret;

  if (!queue->live)
    return -ESHUTDOWN;
  if (request->dma_buf_fd < 0 || !request->bytes ||
      (request->flags & ~NTA_NVME_IMPORT_REQUIRE_CONTIGUOUS))
    return -EINVAL;
  mapping = kzalloc(sizeof(*mapping), GFP_KERNEL);
  if (!mapping)
    return -ENOMEM;
  mapping->dmabuf = dma_buf_get(request->dma_buf_fd);
  if (IS_ERR(mapping->dmabuf)) {
    ret = PTR_ERR(mapping->dmabuf);
    mapping->dmabuf = NULL;
    goto fail;
  }
  if (request->bytes > mapping->dmabuf->size) {
    ret = -EINVAL;
    goto fail;
  }

  /* Static attachment pins storage and waits the exporter's reservation
   * fences. peer2peer remains false, so unsupported GPU P2P mappings fail or
   * migrate instead of bypassing topology validation. */
  mapping->attachment = dma_buf_attach(mapping->dmabuf, &queue->dev->pdev->dev);
  if (IS_ERR(mapping->attachment)) {
    ret = PTR_ERR(mapping->attachment);
    mapping->attachment = NULL;
    goto fail;
  }
  mapping->sgt =
      dma_buf_map_attachment_unlocked(mapping->attachment, DMA_BIDIRECTIONAL);
  if (IS_ERR(mapping->sgt)) {
    ret = PTR_ERR(mapping->sgt);
    mapping->sgt = NULL;
    goto fail;
  }
  ret = nta_add_mapping_fence(mapping);
  if (ret)
    goto fail;

  request->dma_segments = mapping->sgt->nents;
  request->dma_address = sg_dma_address(mapping->sgt->sgl);
  for_each_sgtable_dma_sg(mapping->sgt, sg, index) {
    if (index > 0 && sg_dma_address(sg) != request->dma_address + total &&
        (request->flags & NTA_NVME_IMPORT_REQUIRE_CONTIGUOUS)) {
      ret = -ERANGE;
      goto fail;
    }
    total += sg_dma_len(sg);
  }
  if (total < request->bytes) {
    ret = -ERANGE;
    goto fail;
  }
  ret = nta_validate_prp_mapping(mapping, queue->dev->controller_page_size,
                                 request->bytes);
  if (ret)
    goto fail;

  mapping->handle = ++queue->next_mapping;
  ret = xa_insert(&queue->mappings, mapping->handle, mapping, GFP_KERNEL);
  if (ret)
    goto fail;
  request->handle = mapping->handle;
  request->mapped_bytes = total;
  return 0;

fail:
  nta_release_mapping(queue, mapping);
  return ret;
}

static long nta_ioctl(struct file *file, unsigned int command,
                      unsigned long argument) {
  struct nta_nvme_queue *queue = file->private_data;
  struct nta_nvme_dev *dev = queue->dev;
  void __user *user = (void __user *)argument;
  long ret = 0;

  if (_IOC_TYPE(command) != NTA_NVME_IOCTL_MAGIC)
    return -ENOTTY;
  mutex_lock(&dev->lock);
  if (dev->removing) {
    ret = -ENODEV;
    goto out;
  }
  if (dev->fatal && command != NTA_NVME_IOCTL_GET_INFO &&
      command != NTA_NVME_IOCTL_QUIESCE) {
    ret = -EIO;
    goto out;
  }

  switch (command) {
  case NTA_NVME_IOCTL_GET_INFO: {
    struct nta_nvme_info info = {
        .abi_version = NTA_NVME_ABI_VERSION,
        .queue_depth = NTA_IO_DEPTH,
        .controller_page_size = dev->controller_page_size,
        .lba_shift = dev->lba_shift,
        .namespace_id = target_nsid,
        .doorbell_stride = dev->doorbell_stride,
        .max_transfer_bytes = dev->max_transfer_bytes,
        .capabilities = NTA_NVME_CAP_IOMMU_TRANSLATED |
                        NTA_NVME_CAP_NAMESPACE_READ_ONLY |
                        NTA_NVME_CAP_STATIC_DMA_BUF |
                        NTA_NVME_CAP_MULTI_QUEUE |
                        NTA_NVME_CAP_TRUSTED_RAW_QUEUE,
        .queue_id = queue->qid,
        .queue_count = dev->queue_count,
        .generation = dev->generation,
        .namespace_blocks = dev->namespace_blocks,
        .queue_bytes = queue->bytes,
        .control_offset = queue->control_offset,
        .sq_offset = queue->sq_offset,
        .cq_offset = queue->cq_offset,
        .prp_offset = queue->prp_offset,
        .prp_dma_address = queue->memory_dma + queue->prp_offset,
        .sq_doorbell_offset = 2U * queue->qid * dev->doorbell_stride,
        .cq_doorbell_offset = (2U * queue->qid + 1U) * dev->doorbell_stride,
        .doorbell_mmap_bytes = PAGE_SIZE,
    };
    if (copy_to_user(user, &info, sizeof(info)))
      ret = -EFAULT;
    break;
  }
  case NTA_NVME_IOCTL_IMPORT_DMA_BUF: {
    struct nta_nvme_import request;
    if (copy_from_user(&request, user, sizeof(request))) {
      ret = -EFAULT;
      break;
    }
    ret = nta_import_dmabuf(queue, &request);
    if (!ret && copy_to_user(user, &request, sizeof(request))) {
      struct nta_dma_mapping *mapping =
          xa_erase(&queue->mappings, request.handle);
      (void)nta_delete_io_queue_locked(queue);
      nta_release_mapping(queue, mapping);
      ret = -EFAULT;
    }
    break;
  }
  case NTA_NVME_IOCTL_RELEASE_DMA_BUF: {
    struct nta_nvme_release request;
    struct nta_dma_mapping *mapping;
    if (copy_from_user(&request, user, sizeof(request))) {
      ret = -EFAULT;
      break;
    }
    ret = nta_delete_io_queue_locked(queue);
    if (ret && !dev->fatal)
      break;
    mapping = xa_erase(&queue->mappings, request.handle);
    if (!mapping) {
      ret = -ENOENT;
      break;
    }
    nta_release_mapping(queue, mapping);
    break;
  }
  case NTA_NVME_IOCTL_REGISTER_HOST: {
    struct nta_nvme_register_host request;
    if (copy_from_user(&request, user, sizeof(request))) {
      ret = -EFAULT;
      break;
    }
    ret = nta_register_host(queue, &request);
    if (!ret && copy_to_user(user, &request, sizeof(request))) {
      struct nta_dma_mapping *mapping =
          xa_erase(&queue->mappings, request.handle);
      (void)nta_delete_io_queue_locked(queue);
      nta_release_mapping(queue, mapping);
      ret = -EFAULT;
    }
    break;
  }
  case NTA_NVME_IOCTL_GET_DMA_PAGES: {
    struct nta_nvme_dma_pages *request = memdup_user(user, sizeof(*request));
    if (IS_ERR(request)) {
      ret = PTR_ERR(request);
      break;
    }
    ret = nta_dma_pages(queue, request);
    if (!ret && copy_to_user(user, request, sizeof(*request)))
      ret = -EFAULT;
    kfree(request);
    break;
  }
  case NTA_NVME_IOCTL_QUIESCE:
    ret = nta_delete_io_queue_locked(queue);
    break;
  default:
    ret = -ENOTTY;
  }

out:
  mutex_unlock(&dev->lock);
  return ret;
}

static int nta_open(struct inode *inode, struct file *file) {
  struct miscdevice *misc = file->private_data;
  struct nta_nvme_dev *dev = container_of(misc, struct nta_nvme_dev, misc);
  struct nta_nvme_queue *queue;
  unsigned long qid;
  int ret;

  if (!capable(CAP_SYS_RAWIO))
    return -EPERM;
  mutex_lock(&dev->lock);
  if (dev->removing || dev->suspended) {
    ret = -ENODEV;
    goto out_unlock;
  }
  if (dev->fatal) {
    ret = nta_recover_controller_locked(dev);
    if (ret)
      goto out_unlock;
  } else if (!dev->open_count) {
    ret = nta_identify(dev);
    if (!ret)
      ret = nta_protect_active_namespaces(dev);
    if (ret) {
      nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
      goto out_unlock;
    }
  }

  qid = find_next_zero_bit(dev->queue_bitmap, dev->queue_count + 1U, 1U);
  if (qid > dev->queue_count) {
    ret = -EBUSY;
    goto out_unlock;
  }
  __set_bit(qid, dev->queue_bitmap);
  queue = kzalloc(sizeof(*queue), GFP_KERNEL);
  if (!queue) {
    ret = -ENOMEM;
    goto out_clear_bit;
  }
  queue->dev = dev;
  queue->qid = qid;
  queue->control_offset = 0;
  queue->sq_offset = PAGE_SIZE;
  queue->cq_offset = 2U * PAGE_SIZE;
  queue->prp_offset = 3U * PAGE_SIZE;
  queue->bytes = queue->prp_offset + NTA_IO_DEPTH * PAGE_SIZE;
  INIT_LIST_HEAD(&queue->node);
  kref_init(&queue->refs);
  xa_init(&queue->mappings);
  queue->memory = dma_alloc_coherent(&dev->pdev->dev, queue->bytes,
                                     &queue->memory_dma, GFP_KERNEL);
  if (!queue->memory) {
    ret = -ENOMEM;
    goto out_free_queue;
  }
  memset(queue->memory, 0, queue->bytes);
  queue->control = queue->memory;
  queue->control->magic = NTA_NVME_QUEUE_CONTROL_MAGIC;
  queue->control->abi_version = NTA_NVME_ABI_VERSION;
  queue->control->generation = dev->generation;
  queue->control->queue_id = queue->qid;
  queue->control->state = NTA_NVME_QUEUE_OFFLINE;

  ret = nta_create_io_queue_locked(queue);
  if (ret)
    goto out_free_dma;
  list_add_tail(&queue->node, &dev->queues);
  ++dev->open_count;
  atomic_inc(&dev->queue_refs);
  file->private_data = queue;
  mutex_unlock(&dev->lock);
  return 0;

out_free_dma:
  dma_free_coherent(&dev->pdev->dev, queue->bytes, queue->memory,
                    queue->memory_dma);
out_free_queue:
  xa_destroy(&queue->mappings);
  kfree(queue);
out_clear_bit:
  __clear_bit(qid, dev->queue_bitmap);
out_unlock:
  mutex_unlock(&dev->lock);
  return ret;
}

static int nta_release(struct inode *inode, struct file *file) {
  struct nta_nvme_queue *queue = file->private_data;
  struct nta_nvme_dev *dev;

  if (!queue)
    return 0;
  dev = queue->dev;
  mutex_lock(&dev->lock);
  (void)nta_delete_io_queue_locked(queue);
  nta_release_all_mappings(queue);
  if (!queue->detached) {
    list_del_init(&queue->node);
    __clear_bit(queue->qid, dev->queue_bitmap);
    queue->detached = true;
    --dev->open_count;
  }
  mutex_unlock(&dev->lock);
  file->private_data = NULL;
  kref_put(&queue->refs, nta_queue_free);
  return 0;
}

static void nta_vma_open(struct vm_area_struct *vma) {
  struct nta_nvme_queue *queue = vma->vm_private_data;

  kref_get(&queue->refs);
}

static void nta_vma_close(struct vm_area_struct *vma) {
  struct nta_nvme_queue *queue = vma->vm_private_data;

  kref_put(&queue->refs, nta_queue_free);
}

static const struct vm_operations_struct nta_vm_ops = {
    .open = nta_vma_open,
    .close = nta_vma_close,
};

static int nta_mmap(struct file *file, struct vm_area_struct *vma) {
  struct nta_nvme_queue *queue = file->private_data;
  struct nta_nvme_dev *dev = queue->dev;
  unsigned long selector = vma->vm_pgoff;
  size_t bytes = vma->vm_end - vma->vm_start;
  int ret;

  if (READ_ONCE(dev->removing))
    return -ENODEV;
  vma->vm_pgoff = 0;
  switch (selector) {
  case NTA_NVME_MMAP_QUEUE_PGOFF:
    if (bytes != queue->bytes)
      return -EINVAL;
    ret = dma_mmap_coherent(&dev->pdev->dev, vma, queue->memory,
                            queue->memory_dma, queue->bytes);
    break;
  case NTA_NVME_MMAP_DOORBELL_PGOFF:
    if (bytes != PAGE_SIZE)
      return -EINVAL;
    vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    ret = remap_pfn_range(vma, vma->vm_start,
                          (dev->bar_start + NVME_REG_DBS) >> PAGE_SHIFT,
                          PAGE_SIZE, vma->vm_page_prot);
    break;
  default:
    return -EINVAL;
  }
  if (!ret) {
    vma->vm_private_data = queue;
    vma->vm_ops = &nta_vm_ops;
  }
  return ret;
}

static const struct file_operations nta_fops = {
    .owner = THIS_MODULE,
    .open = nta_open,
    .release = nta_release,
    .unlocked_ioctl = nta_ioctl,
#ifdef CONFIG_COMPAT
    .compat_ioctl = nta_ioctl,
#endif
    .mmap = nta_mmap,
    .llseek = noop_llseek,
};

static bool nta_matches_target(struct pci_dev *pdev) {
  char bdf[16];

  snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%u", pci_domain_nr(pdev->bus),
           pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
  return strcmp(bdf, target_bdf) == 0;
}

struct nta_iommu_group_check {
  struct device *target;
  u32 devices;
  bool foreign;
};

static int nta_check_group_device(struct device *device, void *data) {
  struct nta_iommu_group_check *check = data;

  ++check->devices;
  if (device != check->target)
    check->foreign = true;
  return 0;
}

static int nta_require_isolated_iommu(struct pci_dev *pdev) {
  struct nta_iommu_group_check check = {.target = &pdev->dev};
  struct iommu_domain *domain = iommu_get_domain_for_dev(&pdev->dev);
  struct iommu_group *group;

  if (!domain || (domain->type & IOMMU_DOMAIN_DMA) != IOMMU_DOMAIN_DMA) {
    dev_err(&pdev->dev, "translated DMA IOMMU domain is required\n");
    return -EACCES;
  }
  group = iommu_group_get(&pdev->dev);
  if (!group)
    return -EACCES;
  iommu_group_for_each_dev(group, &check, nta_check_group_device);
  iommu_group_put(group);
  if (check.foreign || check.devices != 1U) {
    dev_err(&pdev->dev, "device must be alone in its IOMMU group\n");
    return -EACCES;
  }
  return 0;
}

static int nta_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
  struct nta_nvme_dev *dev;
  u32 min_page_shift;
  int ret;

  if (!nta_matches_target(pdev))
    return -ENODEV;
  if (PAGE_SIZE != SZ_4K) {
    dev_err(&pdev->dev, "4 KiB host pages are required for doorbell isolation\n");
    return -EOPNOTSUPP;
  }
  ret = nta_require_isolated_iommu(pdev);
  if (ret)
    return ret;

  dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
  if (!dev)
    return -ENOMEM;
  dev->pdev = pdev;
  dev->controller_page_size = PAGE_SIZE;
  dev->generation = 1;
  mutex_init(&dev->lock);
  INIT_LIST_HEAD(&dev->queues);
  atomic_set(&dev->queue_refs, 0);
  init_waitqueue_head(&dev->close_wait);
  pci_set_drvdata(pdev, dev);

  ret = pci_enable_device_mem(pdev);
  if (ret)
    return ret;
  ret = pci_request_mem_regions(pdev, NTA_DRIVER_NAME);
  if (ret)
    goto fail_disable;
  ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
  if (ret)
    goto fail_regions;
  pci_set_master(pdev);
  dev->bar_start = pci_resource_start(pdev, 0);
  if (!PAGE_ALIGNED(dev->bar_start + NVME_REG_DBS) ||
      pci_resource_len(pdev, 0) < NVME_REG_DBS + PAGE_SIZE) {
    ret = -ENOSPC;
    goto fail_master;
  }
  dev->bar = pci_iomap(pdev, 0, 0);
  if (!dev->bar) {
    ret = -ENOMEM;
    goto fail_master;
  }
  dev->cap = readq(dev->bar + NVME_REG_CAP);
  dev->doorbell_stride = 4U << NVME_CAP_STRIDE(dev->cap);
  min_page_shift = 12U + NVME_CAP_MPSMIN(dev->cap);
  if (min_page_shift > PAGE_SHIFT ||
      PAGE_SHIFT > 12U + NVME_CAP_MPSMAX(dev->cap) ||
      NTA_ADMIN_DEPTH - 1U > NVME_CAP_MQES(dev->cap) ||
      NTA_IO_DEPTH - 1U > NVME_CAP_MQES(dev->cap)) {
    ret = -EOPNOTSUPP;
    goto fail_iounmap;
  }

  dev->admin_sq =
      dma_alloc_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_sq),
                         &dev->admin_sq_dma, GFP_KERNEL);
  dev->admin_cq =
      dma_alloc_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_cq),
                         &dev->admin_cq_dma, GFP_KERNEL);
  if (!dev->admin_sq || !dev->admin_cq) {
    ret = -ENOMEM;
    goto fail_dma;
  }

  mutex_lock(&dev->lock);
  ret = nta_start_controller_locked(dev);
  mutex_unlock(&dev->lock);
  if (ret)
    goto fail_dma;

  dev->misc.minor = MISC_DYNAMIC_MINOR;
  dev->misc.name = "nta_nvme";
  dev->misc.fops = &nta_fops;
  dev->misc.parent = &pdev->dev;
  dev->misc.mode = 0600;
  ret = misc_register(&dev->misc);
  if (ret)
    goto fail_controller;

  dev_info(&pdev->dev,
           "trusted read-only GPU queues ready: queues=%u depth=%u nsid=%u\n",
           dev->queue_count, NTA_IO_DEPTH, target_nsid);
  return 0;

fail_controller:
  mutex_lock(&dev->lock);
  (void)nta_disable_controller(dev);
  mutex_unlock(&dev->lock);
fail_dma:
  bitmap_free(dev->queue_bitmap);
  if (dev->admin_cq)
    dma_free_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_cq),
                      dev->admin_cq, dev->admin_cq_dma);
  if (dev->admin_sq)
    dma_free_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_sq),
                      dev->admin_sq, dev->admin_sq_dma);
fail_iounmap:
  pci_iounmap(pdev, dev->bar);
fail_master:
  pci_clear_master(pdev);
fail_regions:
  pci_release_mem_regions(pdev);
fail_disable:
  pci_disable_device(pdev);
  return ret;
}

static void nta_remove(struct pci_dev *pdev) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);

  mutex_lock(&dev->lock);
  dev->removing = true;
  nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_REMOVED);
  mutex_unlock(&dev->lock);
  misc_deregister(&dev->misc);

  wait_event(dev->close_wait, atomic_read(&dev->queue_refs) == 0);
  bitmap_free(dev->queue_bitmap);
  dma_free_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_cq),
                    dev->admin_cq, dev->admin_cq_dma);
  dma_free_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_sq),
                    dev->admin_sq, dev->admin_sq_dma);
  pci_iounmap(pdev, dev->bar);
  pci_clear_master(pdev);
  pci_release_mem_regions(pdev);
  pci_disable_device(pdev);
}

static void nta_shutdown(struct pci_dev *pdev) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);

  mutex_lock(&dev->lock);
  nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_REMOVED);
  mutex_unlock(&dev->lock);
}

static pci_ers_result_t nta_error_detected(struct pci_dev *pdev,
                                           pci_channel_state_t state) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);

  mutex_lock(&dev->lock);
  nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
  mutex_unlock(&dev->lock);
  return state == pci_channel_io_perm_failure ? PCI_ERS_RESULT_DISCONNECT
                                               : PCI_ERS_RESULT_NEED_RESET;
}

static pci_ers_result_t nta_slot_reset(struct pci_dev *pdev) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);
  int ret;

  mutex_lock(&dev->lock);
  if (dev->open_count || atomic_read(&dev->queue_refs)) {
    mutex_unlock(&dev->lock);
    return PCI_ERS_RESULT_DISCONNECT;
  }
  pci_restore_state(pdev);
  ret = nta_start_controller_locked(dev);
  mutex_unlock(&dev->lock);
  return ret ? PCI_ERS_RESULT_DISCONNECT : PCI_ERS_RESULT_RECOVERED;
}

static void nta_reset_prepare(struct pci_dev *pdev) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);

  mutex_lock(&dev->lock);
  nta_mark_fatal_locked(dev, NTA_NVME_QUEUE_FATAL);
  mutex_unlock(&dev->lock);
}

static void nta_reset_done(struct pci_dev *pdev) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);

  mutex_lock(&dev->lock);
  if (!dev->open_count && !atomic_read(&dev->queue_refs))
    (void)nta_start_controller_locked(dev);
  mutex_unlock(&dev->lock);
}

static const struct pci_error_handlers nta_error_handlers = {
    .error_detected = nta_error_detected,
    .slot_reset = nta_slot_reset,
    .reset_prepare = nta_reset_prepare,
    .reset_done = nta_reset_done,
};

static int nta_suspend(struct device *device) {
  struct pci_dev *pdev = to_pci_dev(device);
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);
  int ret = 0;

  mutex_lock(&dev->lock);
  if (dev->open_count || atomic_read(&dev->queue_refs)) {
    ret = -EBUSY;
  } else {
    ret = nta_disable_controller(dev);
    if (!ret) {
      dev->suspended = true;
      pci_clear_master(pdev);
    }
  }
  mutex_unlock(&dev->lock);
  return ret;
}

static int nta_resume(struct device *device) {
  struct pci_dev *pdev = to_pci_dev(device);
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);
  int ret;

  mutex_lock(&dev->lock);
  dev->suspended = false;
  ++dev->generation;
  ret = nta_start_controller_locked(dev);
  mutex_unlock(&dev->lock);
  return ret;
}

static DEFINE_SIMPLE_DEV_PM_OPS(nta_pm_ops, nta_suspend, nta_resume);

static const struct pci_device_id nta_pci_ids[] = {
    {PCI_DEVICE(0x1e0f, 0x002c)},
    {0},
};
MODULE_DEVICE_TABLE(pci, nta_pci_ids);

static struct pci_driver nta_pci_driver = {
    .name = NTA_DRIVER_NAME,
    .id_table = nta_pci_ids,
    .probe = nta_probe,
    .remove = nta_remove,
    .shutdown = nta_shutdown,
    .err_handler = &nta_error_handlers,
    .driver.pm = pm_sleep_ptr(&nta_pm_ops),
};
module_pci_driver(nta_pci_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Nonresident Acquisition project");
MODULE_DESCRIPTION("Contained read-only NVMe queues for trusted finite GPU kernels");
MODULE_IMPORT_NS("DMA_BUF");
