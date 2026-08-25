// SPDX-License-Identifier: GPL-2.0-only

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/iommu.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "nta/NvmeP2pUapi.h"
#include "nv-p2p.h"

#define NTA_NVME_P2P_NAME "nta_nvme_p2p"
#define NTA_NVME_P2P_ALIGNMENT (64ULL * 1024ULL)

struct nta_p2p_mapping {
	struct list_head node;
	u64 handle;
	u64 gpu_address;
	u64 bytes;
	struct pci_dev *peer;
	struct nvidia_p2p_page_table *page_table;
	struct nvidia_p2p_dma_mapping *dma_mapping;
	u32 iommu_entries;
};

struct nta_p2p_file {
	struct mutex lock;
	struct list_head mappings;
	u64 next_handle;
};

static dev_t nta_device_number;
static struct cdev nta_cdev;
static struct class *nta_class;

static u32 nta_page_size(enum nvidia_p2p_page_size_type type)
{
	switch (type) {
	case NVIDIA_P2P_PAGE_SIZE_4KB:
		return 4U * 1024U;
	case NVIDIA_P2P_PAGE_SIZE_64KB:
		return 64U * 1024U;
	case NVIDIA_P2P_PAGE_SIZE_128KB:
		return 128U * 1024U;
	default:
		return 0;
	}
}

static void nta_unmap_peer_iovas(struct nta_p2p_mapping *mapping)
{
	struct iommu_domain *domain;
	u32 page_size;
	u32 index;

	if (!mapping->peer || !mapping->dma_mapping || !mapping->iommu_entries)
		return;
	domain = iommu_get_domain_for_dev(&mapping->peer->dev);
	page_size = nta_page_size(mapping->dma_mapping->page_size_type);
	if (!domain || domain->cookie_type != IOMMU_COOKIE_IOMMUFD || !page_size)
		return;
	for (index = 0; index < mapping->iommu_entries; ++index) {
		dma_addr_t iova = mapping->dma_mapping->dma_addresses[index];

		if (iommu_iova_to_phys(domain, iova) == iova)
			iommu_unmap(domain, iova, page_size);
	}
	mapping->iommu_entries = 0;
}

static int nta_map_peer_iovas(struct nta_p2p_mapping *mapping)
{
	struct nvidia_p2p_dma_mapping *dma = mapping->dma_mapping;
	struct iommu_domain *domain;
	u32 page_size = nta_page_size(dma->page_size_type);
	u32 index;
	int status;

	domain = iommu_get_domain_for_dev(&mapping->peer->dev);
	if (!domain || !(domain->type & __IOMMU_DOMAIN_PAGING) ||
	    domain->cookie_type != IOMMU_COOKIE_IOMMUFD)
		return -EXDEV;
	for (index = 0; index < dma->entries; ++index) {
		dma_addr_t iova = dma->dma_addresses[index];

		if (!IS_ALIGNED(iova, page_size) ||
		    iova > ULONG_MAX - (page_size - 1U) ||
		    (domain->geometry.force_aperture &&
		     (iova < domain->geometry.aperture_start ||
		      iova + page_size - 1U > domain->geometry.aperture_end)) ||
		    iommu_iova_to_phys(domain, iova) != 0) {
			status = -ERANGE;
			goto rollback;
		}
		/*
		 * NVIDIA returns the PCI peer bus address. An IOMMUFD paging domain
		 * does not consume the DMA-API mapping, so install the identity PTE in
		 * that exact domain and retain ownership until the file mapping dies.
		 */
		status = iommu_map(domain, iova, iova, page_size,
				   IOMMU_READ | IOMMU_WRITE, GFP_KERNEL);
		if (status)
			goto rollback;
		mapping->iommu_entries = index + 1U;
		if (iommu_iova_to_phys(domain, iova) != iova) {
			status = -EIO;
			goto rollback;
		}
	}
	pr_info_once("installed peer PTEs in an IOMMUFD paging domain\n");
	return 0;

rollback:
	pr_warn_ratelimited("%s: IOMMUFD peer PTE install failed at %u/%u: %d\n",
			    pci_name(mapping->peer), index, dma->entries, status);
	nta_unmap_peer_iovas(mapping);
	return status;
}

static void nta_release_mapping(struct nta_p2p_mapping *mapping)
{
	if (!mapping)
		return;
	nta_unmap_peer_iovas(mapping);
	if (mapping->dma_mapping)
		nvidia_p2p_dma_unmap_pages(mapping->peer, mapping->page_table,
					   mapping->dma_mapping);
	if (mapping->page_table)
		nvidia_p2p_put_pages_persistent(mapping->gpu_address,
						mapping->page_table, 0);
	if (mapping->peer)
		pci_dev_put(mapping->peer);
	kfree(mapping);
}

static bool nta_peer_is_vfio_nvme(struct pci_dev *peer)
{
	struct device_driver *driver;
	u32 class = peer->class;

	/* pci_dev::class changed from class+revision to the 24-bit class code. */
	if (class != PCI_CLASS_STORAGE_EXPRESS &&
	    (class >> 8) != PCI_CLASS_STORAGE_EXPRESS) {
		pr_warn_ratelimited("rejecting %s: PCI class %#x is not NVMe\n",
				    pci_name(peer), peer->class);
		return false;
	}
	driver = READ_ONCE(peer->dev.driver);
	if (!driver || strcmp(driver->name, "vfio-pci")) {
		pr_warn_ratelimited("rejecting %s: driver is %s, not vfio-pci\n",
				    pci_name(peer),
				    driver ? driver->name : "unbound");
		return false;
	}
	return true;
}

static int nta_validate_map(const struct nta_nvme_p2p_map *request)
{
	if (request->size != sizeof(*request) ||
	    request->abi_version != NTA_NVME_P2P_ABI_VERSION ||
	    request->reserved || request->page_size || request->entry_count ||
	    request->handle)
		return -EINVAL;
	if (!request->gpu_address || !request->bytes ||
	    !IS_ALIGNED(request->gpu_address, NTA_NVME_P2P_ALIGNMENT) ||
	    !IS_ALIGNED(request->bytes, NTA_NVME_P2P_ALIGNMENT))
		return -EINVAL;
	if (request->pci_domain > U16_MAX || request->pci_bus > U8_MAX ||
	    request->pci_device > 31 || request->pci_function > 7)
		return -EINVAL;
	if (!request->dma_addresses || !request->dma_capacity)
		return -EINVAL;
	return 0;
}

static long nta_map(struct nta_p2p_file *context, void __user *argument)
{
	struct nta_nvme_p2p_map request;
	struct nta_p2p_mapping *mapping = NULL;
	struct nvidia_p2p_dma_mapping *dma;
	void __user *addresses;
	u64 covered_bytes;
	u32 page_size;
	long status;

	if (copy_from_user(&request, argument, sizeof(request)))
		return -EFAULT;
	status = nta_validate_map(&request);
	if (status)
		return status;

	mapping = kzalloc(sizeof(*mapping), GFP_KERNEL);
	if (!mapping)
		return -ENOMEM;
	mapping->gpu_address = request.gpu_address;
	mapping->bytes = request.bytes;
	mapping->peer = pci_get_domain_bus_and_slot(
		request.pci_domain, request.pci_bus,
		PCI_DEVFN(request.pci_device, request.pci_function));
	if (!mapping->peer) {
		status = -ENODEV;
		goto fail;
	}
	if (!nta_peer_is_vfio_nvme(mapping->peer)) {
		status = -EPERM;
		goto fail;
	}

	status = nvidia_p2p_get_pages_persistent(
		mapping->gpu_address, mapping->bytes, &mapping->page_table,
		NVIDIA_P2P_FLAGS_DEFAULT);
	if (status) {
		pr_warn_ratelimited("%s: persistent get-pages failed: %ld\n",
				    pci_name(mapping->peer), status);
		goto fail;
	}
	if (!mapping->page_table ||
	    !NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(mapping->page_table)) {
		status = -EPROTO;
		goto fail;
	}

	status = nvidia_p2p_dma_map_pages(mapping->peer, mapping->page_table,
					  &mapping->dma_mapping);
	if (status) {
		pr_warn_ratelimited("%s: peer DMA map failed: %ld\n",
				    pci_name(mapping->peer), status);
		goto fail;
	}
	dma = mapping->dma_mapping;
	if (!dma || !NVIDIA_P2P_DMA_MAPPING_VERSION_COMPATIBLE(dma) ||
	    !dma->dma_addresses || !dma->entries) {
		status = -EPROTO;
		goto fail;
	}
	page_size = nta_page_size(dma->page_size_type);
	if (!page_size || check_mul_overflow((u64)dma->entries,
					      (u64)page_size, &covered_bytes) ||
	    covered_bytes != mapping->bytes) {
		status = -EPROTO;
		goto fail;
	}
	if (request.dma_capacity < dma->entries) {
		status = -ENOSPC;
		goto fail;
	}
	status = nta_map_peer_iovas(mapping);
	if (status)
		goto fail;

	addresses = u64_to_user_ptr(request.dma_addresses);
	if (copy_to_user(addresses, dma->dma_addresses,
			 dma->entries * sizeof(*dma->dma_addresses))) {
		status = -EFAULT;
		goto fail;
	}

	mutex_lock(&context->lock);
	if (!context->next_handle) {
		mutex_unlock(&context->lock);
		status = -EOVERFLOW;
		goto fail;
	}
	mapping->handle = context->next_handle++;
	request.page_size = page_size;
	request.entry_count = dma->entries;
	request.handle = mapping->handle;
	if (copy_to_user(argument, &request, sizeof(request))) {
		mutex_unlock(&context->lock);
		status = -EFAULT;
		goto fail;
	}
	list_add_tail(&mapping->node, &context->mappings);
	mutex_unlock(&context->lock);
	return 0;

fail:
	nta_release_mapping(mapping);
	return status;
}

static long nta_unmap(struct nta_p2p_file *context, void __user *argument)
{
	struct nta_nvme_p2p_unmap request;
	struct nta_p2p_mapping *mapping;
	struct nta_p2p_mapping *found = NULL;

	if (copy_from_user(&request, argument, sizeof(request)))
		return -EFAULT;
	if (request.size != sizeof(request) ||
	    request.abi_version != NTA_NVME_P2P_ABI_VERSION || !request.handle)
		return -EINVAL;

	mutex_lock(&context->lock);
	list_for_each_entry(mapping, &context->mappings, node) {
		if (mapping->handle == request.handle) {
			list_del(&mapping->node);
			found = mapping;
			break;
		}
	}
	mutex_unlock(&context->lock);
	if (!found)
		return -ENOENT;
	nta_release_mapping(found);
	return 0;
}

static long nta_ioctl(struct file *file, unsigned int command,
		      unsigned long argument)
{
	struct nta_p2p_file *context = file->private_data;
	void __user *pointer = (void __user *)argument;

	if (_IOC_TYPE(command) != NTA_NVME_P2P_IOCTL_MAGIC)
		return -ENOTTY;
	switch (command) {
	case NTA_NVME_P2P_IOCTL_MAP:
		return nta_map(context, pointer);
	case NTA_NVME_P2P_IOCTL_UNMAP:
		return nta_unmap(context, pointer);
	default:
		return -ENOTTY;
	}
}

static int nta_open(struct inode *inode, struct file *file)
{
	struct nta_p2p_file *context;

	context = kzalloc(sizeof(*context), GFP_KERNEL);
	if (!context)
		return -ENOMEM;
	mutex_init(&context->lock);
	INIT_LIST_HEAD(&context->mappings);
	context->next_handle = 1;
	file->private_data = context;
	return 0;
}

static int nta_release(struct inode *inode, struct file *file)
{
	struct nta_p2p_file *context = file->private_data;
	struct nta_p2p_mapping *mapping;
	struct nta_p2p_mapping *next;

	mutex_lock(&context->lock);
	list_for_each_entry_safe(mapping, next, &context->mappings, node) {
		list_del(&mapping->node);
		nta_release_mapping(mapping);
	}
	mutex_unlock(&context->lock);
	kfree(context);
	return 0;
}

static const struct file_operations nta_file_operations = {
	.owner = THIS_MODULE,
	.open = nta_open,
	.release = nta_release,
	.unlocked_ioctl = nta_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = compat_ptr_ioctl,
#endif
	.llseek = noop_llseek,
};

static int __init nta_init(void)
{
	int status;

	status = alloc_chrdev_region(&nta_device_number, 0, 1,
				      NTA_NVME_P2P_NAME);
	if (status)
		return status;
	cdev_init(&nta_cdev, &nta_file_operations);
	status = cdev_add(&nta_cdev, nta_device_number, 1);
	if (status)
		goto unregister;
	nta_class = class_create(NTA_NVME_P2P_NAME);
	if (IS_ERR(nta_class)) {
		status = PTR_ERR(nta_class);
		nta_class = NULL;
		goto delete_cdev;
	}
	if (IS_ERR(device_create(nta_class, NULL, nta_device_number, NULL,
				 NTA_NVME_P2P_NAME))) {
		status = -ENODEV;
		goto destroy_class;
	}
	pr_info("loaded NVIDIA peer-page mapper for VFIO NVMe\n");
	return 0;

destroy_class:
	class_destroy(nta_class);
	nta_class = NULL;
delete_cdev:
	cdev_del(&nta_cdev);
unregister:
	unregister_chrdev_region(nta_device_number, 1);
	return status;
}

static void __exit nta_exit(void)
{
	device_destroy(nta_class, nta_device_number);
	class_destroy(nta_class);
	cdev_del(&nta_cdev);
	unregister_chrdev_region(nta_device_number, 1);
	pr_info("unloaded\n");
}

module_init(nta_init);
module_exit(nta_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("NTA NVIDIA peer-page mapper for VFIO NVMe DMA");
MODULE_AUTHOR("NTA contributors");
