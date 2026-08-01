#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bdf=${NTA_NVME_BDF:-0000:d8:00.0}
device=/sys/bus/pci/devices/$bdf
module=$root_dir/driver/nta_nvme/nta_nvme.ko
reference=${NTA_NVME_REFERENCE:-/tmp/nta-nvme-reference.bin}
nsid=${NTA_NVME_NSID:-1}
queues=${NTA_NVME_QUEUES:-8}

die() {
  printf 'nta-nvme-device: %s\n' "$*" >&2
  exit 1
}

current_driver() {
  if [[ -L $device/driver ]]; then
    basename "$(readlink "$device/driver")"
  else
    printf 'none\n'
  fi
}

require_safe_device() {
  [[ -d $device ]] || die "PCI device $bdf does not exist"
  local block
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    local block_device=/dev/$(basename "$block")
    if findmnt -rn -S "$block_device" >/dev/null; then
      die "$block_device is mounted"
    fi
    if [[ -n $(ls -A "/sys/class/block/$(basename "$block")/holders" 2>/dev/null) ]]; then
      die "$block_device has kernel block holders"
    fi
    if sudo fuser "$block_device" >/dev/null 2>&1; then
      die "$block_device is open by another process"
    fi
  done
}

require_containment() {
  [[ $(getconf PAGESIZE) == 4096 ]] ||
    die "4 KiB host pages are required to map only the NVMe doorbell page"
  [[ -L $device/iommu_group ]] || die "$bdf has no IOMMU group"
  local group
  group=$(readlink -f "$device/iommu_group")
  local type
  type=$(<"$group/type")
  case $type in
  DMA | DMA-FQ) ;;
  *) die "IOMMU group $(basename "$group") is '$type', not translated DMA" ;;
  esac
  local members=("$group"/devices/*)
  [[ ${#members[@]} == 1 && $(basename "${members[0]}") == "$bdf" ]] ||
    die "$bdf is not alone in IOMMU group $(basename "$group")"
  [[ $nsid =~ ^[1-9][0-9]*$ ]] || die "NTA_NVME_NSID must be positive"
  [[ $queues =~ ^[1-9][0-9]*$ && $queues -le 32 ]] ||
    die "NTA_NVME_QUEUES must be between 1 and 32"
}

case ${1:-status} in
status)
  printf 'bdf=%s driver=%s node=%s iommu=%s\n' "$bdf" "$(current_driver)" \
    "$([[ -e /dev/nta_nvme ]] && printf present || printf absent)" \
    "$([[ -L $device/iommu_group ]] && cat "$(readlink -f "$device/iommu_group")/type" || printf absent)"
  ;;
preflight)
  require_safe_device
  require_containment
  printf 'containment preflight passed: bdf=%s nsid=%s queues=%s\n' \
    "$bdf" "$nsid" "$queues"
  ;;
build)
  make -C "$root_dir/driver/nta_nvme"
  ;;
bind)
  if lsmod | awk '{print $1}' | grep -qx vmem_sw; then
    die "vmem_sw is loaded and may own this SSD; unload/disable it before binding"
  fi
  require_safe_device
  require_containment
  make -C "$root_dir/driver/nta_nvme"
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    sudo dd if="/dev/$(basename "$block")" bs=4096 count=512 \
      iflag=direct status=none | dd of="$reference" bs=4096 status=none
    break
  done
  if [[ $(current_driver) != none ]]; then
    printf '%s' "$bdf" | sudo tee "$device/driver/unbind" >/dev/null
  fi
  if lsmod | awk '{print $1}' | grep -qx nta_nvme; then
    sudo rmmod nta_nvme
  fi
  sudo insmod "$module" target_bdf="$bdf" target_nsid="$nsid" \
    max_io_queues="$queues"
  printf '%s' nta_nvme | sudo tee "$device/driver_override" >/dev/null
  printf '%s' "$bdf" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
  [[ -e /dev/nta_nvme ]] || die "driver probe failed; inspect dmesg"
  "$0" status
  ;;
restore)
  require_safe_device
  if [[ $(current_driver) == nta_nvme ]]; then
    printf '%s' "$bdf" | sudo tee "$device/driver/unbind" >/dev/null
  fi
  printf '%s' nvmex | sudo tee "$device/driver_override" >/dev/null
  if [[ -d /sys/bus/pci/drivers/nvmex ]]; then
    printf '%s' "$bdf" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
  else
    printf '%s' nvme | sudo tee "$device/driver_override" >/dev/null
    printf '%s' "$bdf" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
  fi
  sudo rmmod nta_nvme 2>/dev/null || true
  "$0" status
  ;;
*)
  die "usage: $0 {status|preflight|build|bind|restore}"
  ;;
esac
