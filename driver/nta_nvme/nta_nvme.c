// SPDX-License-Identifier: GPL-2.0

#include <linux/delay.h>
#include <linux/dma-buf.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/nvme.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/xarray.h>

#include "nta/NvmeUapi.h"

#define NTA_DRIVER_NAME "nta_nvme"
#define NTA_ADMIN_DEPTH 32U
#define NTA_IO_DEPTH 64U
#define NTA_ADMIN_TIMEOUT_US (10U * 1000U * 1000U)
#define NTA_READY_POLL_US 1000U
#define NTA_QUEUE_ID 1U

static char target_bdf[16] = "0000:d8:00.0";
module_param_string(target_bdf, target_bdf, sizeof(target_bdf), 0444);
MODULE_PARM_DESC(target_bdf,
                 "Only bind the exact PCI domain:bus:slot.function");

struct nta_dma_mapping {
  struct dma_buf *dmabuf;
  struct dma_buf_attachment *attachment;
  struct sg_table *sgt;
  enum dma_data_direction direction;
  struct page **pages;
  unsigned long page_count;
  struct sg_table host_sgt;
  bool host_mapping;
  bool host_sgt_valid;
  bool host_dma_mapped;
  u64 handle;
  bool invalidated;
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
  u64 namespace_blocks;

  struct nvme_command *admin_sq;
  dma_addr_t admin_sq_dma;
  struct nvme_completion *admin_cq;
  dma_addr_t admin_cq_dma;
  u16 admin_tail;
  u16 admin_head;
  u16 admin_cid;
  u8 admin_phase;

  void *queue_memory;
  dma_addr_t queue_dma;
  size_t queue_bytes;
  size_t sq_offset;
  size_t cq_offset;
  size_t prp_offset;

  struct miscdevice misc;
  atomic_t open_count;
  struct mutex ioctl_lock;
  struct xarray mappings;
  u64 next_mapping;
  bool io_queue_live;
};

static void nta_dmabuf_move_notify(struct dma_buf_attachment *attachment) {
  struct nta_dma_mapping *mapping = attachment->importer_priv;

  WRITE_ONCE(mapping->invalidated, true);
}

static const struct dma_buf_attach_ops nta_dmabuf_attach_ops = {
    .allow_peer2peer = true,
    .move_notify = nta_dmabuf_move_notify,
};

static void __iomem *nta_doorbell(struct nta_nvme_dev *dev, u16 qid,
                                  bool completion) {
  u32 index = 2U * qid + (completion ? 1U : 0U);

  return dev->bar + NVME_REG_DBS + index * dev->doorbell_stride;
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

  command->common.command_id = cid;
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
    if (time_after(jiffies, timeout))
      return -ETIMEDOUT;
    usleep_range(50, 100);
  } while (true);
  dma_rmb();

  if (completion->command_id != cid)
    return -EIO;
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
  if (ctrl->mdts && ctrl->mdts < 9U)
    dev->max_transfer_bytes = dev->controller_page_size << ctrl->mdts;
  else
    dev->max_transfer_bytes = dev->controller_page_size * 512U;

  memset(identify, 0, NVME_IDENTIFY_DATA_SIZE);
  memset(&command, 0, sizeof(command));
  command.identify.opcode = nvme_admin_identify;
  command.identify.nsid = cpu_to_le32(1U);
  command.identify.dptr.prp1 = cpu_to_le64(identify_dma);
  command.identify.cns = NVME_ID_CNS_NS;
  ret = nta_admin_command(dev, &command, NULL);
  if (ret)
    goto out;

  ns = identify;
  format = nvme_lbaf_index(ns->flbas);
  if (format > ns->nlbaf || ns->lbaf[format].ds < 9U ||
      ns->lbaf[format].ds > 16U || le16_to_cpu(ns->lbaf[format].ms) != 0U ||
      (ns->dps & NVME_NS_DPS_PI_MASK) != 0U) {
    dev_err(&dev->pdev->dev, "namespace 1 has an unsupported LBA format\n");
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

static int nta_create_io_queue(struct nta_nvme_dev *dev) {
  struct nvme_command command = {};
  u32 result;
  int ret;

  command.features.opcode = nvme_admin_set_features;
  command.features.fid = cpu_to_le32(NVME_FEAT_NUM_QUEUES);
  command.features.dword11 = 0;
  ret = nta_admin_command(dev, &command, &result);
  if (ret)
    return ret;
  if ((result & 0xffffU) == 0xffffU || (result >> 16) == 0xffffU)
    return -ENOSPC;

  memset(&command, 0, sizeof(command));
  command.create_cq.opcode = nvme_admin_create_cq;
  command.create_cq.prp1 = cpu_to_le64(dev->queue_dma + dev->cq_offset);
  command.create_cq.cqid = cpu_to_le16(NTA_QUEUE_ID);
  command.create_cq.qsize = cpu_to_le16(NTA_IO_DEPTH - 1U);
  command.create_cq.cq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
  ret = nta_admin_command(dev, &command, NULL);
  if (ret)
    return ret;

  memset(&command, 0, sizeof(command));
  command.create_sq.opcode = nvme_admin_create_sq;
  command.create_sq.prp1 = cpu_to_le64(dev->queue_dma + dev->sq_offset);
  command.create_sq.sqid = cpu_to_le16(NTA_QUEUE_ID);
  command.create_sq.qsize = cpu_to_le16(NTA_IO_DEPTH - 1U);
  command.create_sq.sq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
  command.create_sq.cqid = cpu_to_le16(NTA_QUEUE_ID);
  ret = nta_admin_command(dev, &command, NULL);
  if (ret) {
    memset(&command, 0, sizeof(command));
    command.delete_queue.opcode = nvme_admin_delete_cq;
    command.delete_queue.qid = cpu_to_le16(NTA_QUEUE_ID);
    nta_admin_command(dev, &command, NULL);
    return ret;
  }
  dev->io_queue_live = true;
  return 0;
}

static void nta_delete_io_queue(struct nta_nvme_dev *dev) {
  struct nvme_command command = {};
  int ret;

  if (!dev->io_queue_live)
    return;
  command.delete_queue.opcode = nvme_admin_delete_sq;
  command.delete_queue.qid = cpu_to_le16(NTA_QUEUE_ID);
  ret = nta_admin_command(dev, &command, NULL);
  memset(&command, 0, sizeof(command));
  command.delete_queue.opcode = nvme_admin_delete_cq;
  command.delete_queue.qid = cpu_to_le16(NTA_QUEUE_ID);
  ret = nta_admin_command(dev, &command, NULL) ?: ret;
  if (ret)
    nta_disable_controller(dev);
  dev->io_queue_live = false;
}

static void nta_release_mapping(struct nta_nvme_dev *dev,
                                struct nta_dma_mapping *mapping) {
  if (mapping->host_mapping && mapping->host_dma_mapped)
    dma_unmap_sgtable(&dev->pdev->dev, &mapping->host_sgt, mapping->direction,
                      0);
  else if (mapping->sgt)
    dma_buf_unmap_attachment_unlocked(mapping->attachment, mapping->sgt,
                                      mapping->direction);
  if (mapping->attachment)
    dma_buf_detach(mapping->dmabuf, mapping->attachment);
  if (mapping->dmabuf)
    dma_buf_put(mapping->dmabuf);
  if (mapping->host_sgt_valid)
    sg_free_table(&mapping->host_sgt);
  if (mapping->pages) {
    unpin_user_pages_dirty_lock(mapping->pages, mapping->page_count,
                                mapping->host_mapping);
    kvfree(mapping->pages);
  }
  kfree(mapping);
}

static void nta_release_all_mappings(struct nta_nvme_dev *dev) {
  struct nta_dma_mapping *mapping;
  unsigned long index;

  xa_for_each(&dev->mappings, index, mapping) {
    xa_erase(&dev->mappings, index);
    nta_release_mapping(dev, mapping);
  }
}

static u32 nta_dma_page_count(const struct nta_dma_mapping *mapping,
                              u32 page_size) {
  struct scatterlist *sg;
  u64 pages = 0;
  unsigned int index;

  for_each_sgtable_dma_sg(mapping->sgt, sg, index) pages +=
      DIV_ROUND_UP(sg_dma_len(sg), page_size);
  return min_t(u64, pages, U32_MAX);
}

static int nta_dma_pages(struct nta_nvme_dev *dev,
                         struct nta_nvme_dma_pages *request) {
  struct nta_dma_mapping *mapping = xa_load(&dev->mappings, request->handle);
  struct scatterlist *sg;
  u32 logical_page = 0;
  u32 copied = 0;
  unsigned int index;

  if (!mapping || READ_ONCE(mapping->invalidated))
    return -ENOENT;
  if (request->page_count == 0 || request->page_count > NTA_NVME_MAX_DMA_PAGES)
    return -EINVAL;

  for_each_sgtable_dma_sg(mapping->sgt, sg, index) {
    dma_addr_t address = sg_dma_address(sg);
    u32 pages = DIV_ROUND_UP(sg_dma_len(sg), dev->controller_page_size);
    u32 page;

    for (page = 0; page < pages; ++page, ++logical_page) {
      if (logical_page < request->first_page)
        continue;
      if (copied == request->page_count)
        goto done;
      request->addresses[copied++] =
          address + (u64)page * dev->controller_page_size;
    }
  }
done:
  request->page_count = copied;
  return copied ? 0 : -ERANGE;
}

static int nta_register_host(struct nta_nvme_dev *dev,
                             struct nta_nvme_register_host *request) {
  struct nta_dma_mapping *mapping;
  unsigned long address = request->user_address;
  unsigned long pages;
  long pinned;
  int ret;

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
  mapping->direction = DMA_FROM_DEVICE;
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
      mapping->page_count = 0;
    }
    ret = pinned < 0 ? pinned : -EFAULT;
    goto fail;
  }
  ret = sg_alloc_table_from_pages(&mapping->host_sgt, mapping->pages, pages, 0,
                                  request->bytes, GFP_KERNEL);
  if (ret)
    goto fail;
  mapping->host_sgt_valid = true;
  ret = dma_map_sgtable(&dev->pdev->dev, &mapping->host_sgt, mapping->direction,
                        0);
  if (ret)
    goto fail;
  mapping->host_dma_mapped = true;
  mapping->sgt = &mapping->host_sgt;
  mapping->handle = ++dev->next_mapping;
  ret = xa_insert(&dev->mappings, mapping->handle, mapping, GFP_KERNEL);
  if (ret)
    goto fail;

  request->handle = mapping->handle;
  request->dma_pages = nta_dma_page_count(mapping, dev->controller_page_size);
  return 0;

fail:
  nta_release_mapping(dev, mapping);
  return ret;
}

static int nta_import_dmabuf(struct nta_nvme_dev *dev,
                             struct nta_nvme_import *request) {
  struct nta_dma_mapping *mapping;
  struct scatterlist *sg;
  u64 total = 0;
  unsigned int index;
  int ret;

  if (request->dma_buf_fd < 0 || request->bytes == 0)
    return -EINVAL;
  mapping = kzalloc(sizeof(*mapping), GFP_KERNEL);
  if (!mapping)
    return -ENOMEM;
  mapping->direction = request->flags & NTA_NVME_IMPORT_BIDIRECTIONAL
                           ? DMA_BIDIRECTIONAL
                           : DMA_FROM_DEVICE;
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
  mapping->attachment = dma_buf_dynamic_attach(mapping->dmabuf, &dev->pdev->dev,
                                               &nta_dmabuf_attach_ops, mapping);
  if (IS_ERR(mapping->attachment)) {
    ret = PTR_ERR(mapping->attachment);
    mapping->attachment = NULL;
    goto fail;
  }
  mapping->sgt =
      dma_buf_map_attachment_unlocked(mapping->attachment, mapping->direction);
  if (IS_ERR(mapping->sgt)) {
    ret = PTR_ERR(mapping->sgt);
    mapping->sgt = NULL;
    goto fail;
  }
  if (READ_ONCE(mapping->invalidated)) {
    ret = -EAGAIN;
    goto fail;
  }

  request->dma_segments = mapping->sgt->nents;
  request->dma_address = sg_dma_address(mapping->sgt->sgl);
  for_each_sgtable_dma_sg(mapping->sgt, sg, index) {
    if (index > 0 && sg_dma_address(sg) != request->dma_address + total &&
        request->flags & NTA_NVME_IMPORT_REQUIRE_CONTIGUOUS) {
      ret = -ERANGE;
      goto fail;
    }
    total += sg_dma_len(sg);
  }
  if (total < request->bytes) {
    ret = -ERANGE;
    goto fail;
  }

  mapping->handle = ++dev->next_mapping;
  ret = xa_insert(&dev->mappings, mapping->handle, mapping, GFP_KERNEL);
  if (ret)
    goto fail;
  request->handle = mapping->handle;
  request->mapped_bytes = total;
  return 0;

fail:
  nta_release_mapping(dev, mapping);
  return ret;
}

static long nta_ioctl(struct file *file, unsigned int command,
                      unsigned long argument) {
  struct miscdevice *misc = file->private_data;
  struct nta_nvme_dev *dev = container_of(misc, struct nta_nvme_dev, misc);
  void __user *user = (void __user *)argument;
  long ret = 0;

  if (_IOC_TYPE(command) != NTA_NVME_IOCTL_MAGIC)
    return -ENOTTY;
  mutex_lock(&dev->ioctl_lock);
  switch (command) {
  case NTA_NVME_IOCTL_GET_INFO: {
    struct nta_nvme_info info = {
        .abi_version = NTA_NVME_ABI_VERSION,
        .queue_depth = NTA_IO_DEPTH,
        .controller_page_size = dev->controller_page_size,
        .lba_shift = dev->lba_shift,
        .namespace_id = 1,
        .doorbell_stride = dev->doorbell_stride,
        .max_transfer_bytes = dev->max_transfer_bytes,
        .namespace_blocks = dev->namespace_blocks,
        .queue_bytes = dev->queue_bytes,
        .queue_dma_address = dev->queue_dma,
        .sq_offset = dev->sq_offset,
        .cq_offset = dev->cq_offset,
        .prp_offset = dev->prp_offset,
        .sq_doorbell_offset = 2U * NTA_QUEUE_ID * dev->doorbell_stride,
        .cq_doorbell_offset = (2U * NTA_QUEUE_ID + 1U) * dev->doorbell_stride,
        .doorbell_mmap_bytes = dev->controller_page_size,
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
    ret = nta_import_dmabuf(dev, &request);
    if (!ret && copy_to_user(user, &request, sizeof(request))) {
      struct nta_dma_mapping *mapping =
          xa_erase(&dev->mappings, request.handle);
      nta_release_mapping(dev, mapping);
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
    mapping = xa_erase(&dev->mappings, request.handle);
    if (!mapping) {
      ret = -ENOENT;
      break;
    }
    nta_release_mapping(dev, mapping);
    break;
  }
  case NTA_NVME_IOCTL_REGISTER_HOST: {
    struct nta_nvme_register_host request;
    if (copy_from_user(&request, user, sizeof(request))) {
      ret = -EFAULT;
      break;
    }
    ret = nta_register_host(dev, &request);
    if (!ret && copy_to_user(user, &request, sizeof(request))) {
      struct nta_dma_mapping *mapping =
          xa_erase(&dev->mappings, request.handle);
      nta_release_mapping(dev, mapping);
      ret = -EFAULT;
    }
    break;
  }
  case NTA_NVME_IOCTL_GET_DMA_PAGES: {
    struct nta_nvme_dma_pages *request;
    request = memdup_user(user, sizeof(*request));
    if (IS_ERR(request)) {
      ret = PTR_ERR(request);
      break;
    }
    ret = nta_dma_pages(dev, request);
    if (!ret && copy_to_user(user, request, sizeof(*request)))
      ret = -EFAULT;
    kfree(request);
    break;
  }
  case NTA_NVME_IOCTL_QUIESCE:
    nta_delete_io_queue(dev);
    break;
  default:
    ret = -ENOTTY;
  }
  mutex_unlock(&dev->ioctl_lock);
  return ret;
}

static int nta_open(struct inode *inode, struct file *file) {
  struct miscdevice *misc = file->private_data;
  struct nta_nvme_dev *dev = container_of(misc, struct nta_nvme_dev, misc);
  int ret;

  if (atomic_cmpxchg(&dev->open_count, 0, 1) != 0)
    return -EBUSY;
  mutex_lock(&dev->ioctl_lock);
  nta_delete_io_queue(dev);
  nta_release_all_mappings(dev);
  memset(dev->queue_memory, 0, dev->queue_bytes);
  ret = 0;
  if (!(readl(dev->bar + NVME_REG_CC) & NVME_CC_ENABLE))
    ret = nta_enable_controller(dev);
  if (!ret)
    ret = nta_create_io_queue(dev);
  mutex_unlock(&dev->ioctl_lock);
  if (ret) {
    atomic_set(&dev->open_count, 0);
    return ret;
  }
  return 0;
}

static int nta_release(struct inode *inode, struct file *file) {
  struct miscdevice *misc = file->private_data;
  struct nta_nvme_dev *dev = container_of(misc, struct nta_nvme_dev, misc);

  mutex_lock(&dev->ioctl_lock);
  nta_delete_io_queue(dev);
  nta_release_all_mappings(dev);
  mutex_unlock(&dev->ioctl_lock);
  atomic_set(&dev->open_count, 0);
  return 0;
}

static int nta_mmap(struct file *file, struct vm_area_struct *vma) {
  struct miscdevice *misc = file->private_data;
  struct nta_nvme_dev *dev = container_of(misc, struct nta_nvme_dev, misc);
  unsigned long selector = vma->vm_pgoff;
  size_t bytes = vma->vm_end - vma->vm_start;

  vma->vm_pgoff = 0;
  switch (selector) {
  case NTA_NVME_MMAP_QUEUE_PGOFF:
    if (bytes != dev->queue_bytes)
      return -EINVAL;
    return dma_mmap_coherent(&dev->pdev->dev, vma, dev->queue_memory,
                             dev->queue_dma, dev->queue_bytes);
  case NTA_NVME_MMAP_DOORBELL_PGOFF:
    if (bytes != dev->controller_page_size)
      return -EINVAL;
    vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    return remap_pfn_range(vma, vma->vm_start,
                           (dev->bar_start + NVME_REG_DBS) >> PAGE_SHIFT, bytes,
                           vma->vm_page_prot);
  default:
    return -EINVAL;
  }
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

static int nta_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
  struct nta_nvme_dev *dev;
  u32 min_page_shift;
  int ret;

  if (!nta_matches_target(pdev))
    return -ENODEV;
  dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
  if (!dev)
    return -ENOMEM;
  dev->pdev = pdev;
  dev->sq_offset = 0;
  dev->cq_offset = PAGE_SIZE;
  dev->prp_offset = 2U * PAGE_SIZE;
  dev->queue_bytes = dev->prp_offset + NTA_IO_DEPTH * PAGE_SIZE;
  dev->next_mapping = 0;
  atomic_set(&dev->open_count, 0);
  mutex_init(&dev->ioctl_lock);
  xa_init(&dev->mappings);
  pci_set_drvdata(pdev, dev);

  ret = pci_enable_device_mem(pdev);
  if (ret)
    goto fail_xarray;
  ret = pci_request_mem_regions(pdev, NTA_DRIVER_NAME);
  if (ret)
    goto fail_disable;
  ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
  if (ret)
    goto fail_regions;
  pci_set_master(pdev);
  dev->bar_start = pci_resource_start(pdev, 0);
  dev->bar = pci_iomap(pdev, 0, 0);
  if (!dev->bar) {
    ret = -ENOMEM;
    goto fail_master;
  }
  dev->cap = readq(dev->bar + NVME_REG_CAP);
  dev->doorbell_stride = 4U << NVME_CAP_STRIDE(dev->cap);
  min_page_shift = 12U + NVME_CAP_MPSMIN(dev->cap);
  if (min_page_shift > PAGE_SHIFT) {
    ret = -EOPNOTSUPP;
    goto fail_iounmap;
  }
  dev->controller_page_size = PAGE_SIZE;
  if (NTA_IO_DEPTH - 1U > NVME_CAP_MQES(dev->cap)) {
    ret = -EOPNOTSUPP;
    goto fail_iounmap;
  }

  dev->admin_sq =
      dma_alloc_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_sq),
                         &dev->admin_sq_dma, GFP_KERNEL);
  dev->admin_cq =
      dma_alloc_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_cq),
                         &dev->admin_cq_dma, GFP_KERNEL);
  dev->queue_memory = dma_alloc_coherent(&pdev->dev, dev->queue_bytes,
                                         &dev->queue_dma, GFP_KERNEL);
  if (!dev->admin_sq || !dev->admin_cq || !dev->queue_memory) {
    ret = -ENOMEM;
    goto fail_dma;
  }
  memset(dev->queue_memory, 0, dev->queue_bytes);

  ret = nta_disable_controller(dev);
  if (ret)
    goto fail_dma;
  ret = nta_enable_controller(dev);
  if (ret)
    goto fail_dma;
  ret = nta_identify(dev);
  if (ret)
    goto fail_controller;
  ret = nta_create_io_queue(dev);
  if (ret)
    goto fail_controller;

  dev->misc.minor = MISC_DYNAMIC_MINOR;
  dev->misc.name = "nta_nvme";
  dev->misc.fops = &nta_fops;
  dev->misc.parent = &pdev->dev;
  dev->misc.mode = 0600;
  ret = misc_register(&dev->misc);
  if (ret)
    goto fail_queue;

  dev_info(&pdev->dev, "GPU queue ready: depth=%u LBA=%u bytes\n", NTA_IO_DEPTH,
           1U << dev->lba_shift);
  return 0;

fail_queue:
  nta_delete_io_queue(dev);
fail_controller:
  nta_disable_controller(dev);
fail_dma:
  if (dev->queue_memory)
    dma_free_coherent(&pdev->dev, dev->queue_bytes, dev->queue_memory,
                      dev->queue_dma);
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
fail_xarray:
  xa_destroy(&dev->mappings);
  return ret;
}

static void nta_remove(struct pci_dev *pdev) {
  struct nta_nvme_dev *dev = pci_get_drvdata(pdev);

  misc_deregister(&dev->misc);
  nta_delete_io_queue(dev);
  nta_release_all_mappings(dev);
  xa_destroy(&dev->mappings);
  nta_disable_controller(dev);
  dma_free_coherent(&pdev->dev, dev->queue_bytes, dev->queue_memory,
                    dev->queue_dma);
  dma_free_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_cq),
                    dev->admin_cq, dev->admin_cq_dma);
  dma_free_coherent(&pdev->dev, NTA_ADMIN_DEPTH * sizeof(*dev->admin_sq),
                    dev->admin_sq, dev->admin_sq_dma);
  pci_iounmap(pdev, dev->bar);
  pci_clear_master(pdev);
  pci_release_mem_regions(pdev);
  pci_disable_device(pdev);
}

static const struct pci_device_id nta_pci_ids[] = {{PCI_DEVICE(0x1e0f, 0x002c)},
                                                   {
                                                       0,
                                                   }};
MODULE_DEVICE_TABLE(pci, nta_pci_ids);

static struct pci_driver nta_pci_driver = {
    .name = NTA_DRIVER_NAME,
    .id_table = nta_pci_ids,
    .probe = nta_probe,
    .remove = nta_remove,
};
module_pci_driver(nta_pci_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Nonresident Acquisition project");
MODULE_DESCRIPTION("NVMe queue bootstrap for trusted finite GPU kernels");
MODULE_IMPORT_NS("DMA_BUF");
