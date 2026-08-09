#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bdf=${NTA_NVME_BDF:-0000:d8:00.0}
device=/sys/bus/pci/devices/$bdf
reference=${NTA_NVME_REFERENCE:-/tmp/nta-nvme-reference.bin}
reference_bytes=${NTA_NVME_REFERENCE_BYTES:-67108864}
nsid=${NTA_NVME_NSID:-1}
depth=${NTA_NVME_QUEUE_DEPTH:-64}
gpu=${NTA_GPU:-0}
media_policy=${NTA_NVME_MEDIA_POLICY:-hardware-write-protect}
probe=${NTA_VFIO_PROBE:-$root_dir/build/nta-vfio-nvme-probe}
benchmark=${NTA_NVME_BENCHMARK:-$root_dir/build/nta-nvme-bench}
state=/run/nta-vfio-${bdf}.driver

die() {
  printf 'nta-vfio-device: %s\n' "$*" >&2
  exit 1
}

current_driver() {
  if [[ -L $device/driver ]]; then
    basename "$(readlink "$device/driver")"
  else
    printf 'none\n'
  fi
}

vfio_cdev() {
  local entry
  for entry in "$device"/vfio-dev/vfio*; do
    [[ -e $entry ]] || continue
    printf '/dev/vfio/devices/%s\n' "$(basename "$entry")"
    return
  done
  printf 'absent\n'
}

require_safe_device() {
  [[ -d $device ]] || die "PCI device $bdf does not exist"
  local block
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    local block_device
    block_device=/dev/$(basename "$block")
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
    die "4 KiB host pages are required to isolate the NVMe doorbell page"
  [[ -c /dev/iommu ]] || die "/dev/iommu is unavailable"
  [[ -L $device/iommu_group ]] || die "$bdf has no IOMMU group"
  local group
  group=$(readlink -f "$device/iommu_group")
  local members=("$group"/devices/*)
  [[ ${#members[@]} == 1 && $(basename "${members[0]}") == "$bdf" ]] ||
    die "$bdf is not alone in IOMMU group $(basename "$group")"
  if [[ -r /sys/module/vfio/parameters/enable_unsafe_noiommu_mode ]]; then
    [[ $(</sys/module/vfio/parameters/enable_unsafe_noiommu_mode) == N ]] ||
      die "VFIO unsafe no-IOMMU mode is enabled"
  fi
  [[ $nsid =~ ^[1-9][0-9]*$ ]] || die "NTA_NVME_NSID must be positive"
  [[ $depth =~ ^[1-9][0-9]*$ && $depth -ge 2 && $depth -le 4096 ]] ||
    die "NTA_NVME_QUEUE_DEPTH must be between 2 and 4096"
  [[ $gpu =~ ^[0-9]+$ ]] || die "NTA_GPU must be a non-negative integer"
  [[ $reference_bytes =~ ^[1-9][0-9]*$ &&
    $((reference_bytes % 4096)) == 0 ]] ||
    die "NTA_NVME_REFERENCE_BYTES must be a positive multiple of 4096"
  [[ $media_policy == hardware-write-protect ||
    $media_policy == trusted-read-only-code ]] ||
    die "NTA_NVME_MEDIA_POLICY must be hardware-write-protect or trusted-read-only-code"
}

capture_reference() {
  local block
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    sudo dd if="/dev/$(basename "$block")" bs=4096 \
      count="$((reference_bytes / 4096))" \
      iflag=direct status=none | dd of="$reference" bs=4096 status=none
    return
  done
  [[ -f $reference ]] ||
    die "no namespace block device exists and reference file is absent"
}

bind_vfio() {
  local vmem_users
  vmem_users=$(lsmod | awk '$1 == "vmem_sw" { print $3 }')
  if [[ -n $vmem_users && $vmem_users -ne 0 ]]; then
    die "vmem_sw has active references; stop its users before binding"
  fi
  require_safe_device
  require_containment
  capture_reference
  sudo modprobe iommufd
  sudo modprobe vfio-pci
  local driver
  driver=$(current_driver)
  printf '%s\n' "$driver" | sudo tee "$state" >/dev/null
  if [[ $driver != none ]]; then
    printf '%s' "$bdf" | sudo tee "$device/driver/unbind" >/dev/null
  fi
  printf '%s' vfio-pci | sudo tee "$device/driver_override" >/dev/null
  printf '%s' "$bdf" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
  [[ $(current_driver) == vfio-pci ]] || die "vfio-pci probe failed"
  local cdev
  cdev=$(vfio_cdev)
  [[ -c $cdev ]] || die "VFIO cdev $cdev is unavailable"
}

case ${1:-status} in
status)
  group_type=absent
  if [[ -L $device/iommu_group ]]; then
    group_type=$(<"$(readlink -f "$device/iommu_group")/type")
  fi
  printf 'bdf=%s driver=%s cdev=%s iommu_group_type=%s\n' \
    "$bdf" "$(current_driver)" "$(vfio_cdev)" "$group_type"
  ;;
preflight)
  require_safe_device
  require_containment
  printf 'VFIO preflight passed: bdf=%s nsid=%s depth=%s gpu=%s media_policy=%s\n' \
    "$bdf" "$nsid" "$depth" "$gpu" "$media_policy"
  ;;
bind)
  bind_vfio
  "$0" status
  ;;
probe)
  [[ $(current_driver) == vfio-pci ]] || die "$bdf is not bound to vfio-pci"
  [[ -x $probe ]] || die "probe executable is absent; build nta-vfio-nvme-probe"
  if [[ ${NTA_VFIO_PROBE_SUDO:-1} == 1 ]]; then
    sudo env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-12.9/lib64}" \
      "$probe" "vfio:$bdf" "$gpu" "$nsid" "$depth" "$media_policy"
  else
    "$probe" "vfio:$bdf" "$gpu" "$nsid" "$depth" "$media_policy"
  fi
  ;;
bind-and-probe)
  bind_vfio
  if ! "$0" probe; then
    printf 'VFIO probe failed; restoring the previous PCI driver\n' >&2
    "$0" restore
    exit 1
  fi
  ;;
qualify)
  bind_vfio
  restore_on_exit() {
    "$0" restore >/dev/null 2>&1 || true
  }
  trap restore_on_exit EXIT
  [[ -x $benchmark ]] || die "benchmark executable is absent; build nta-nvme-bench"
  sudo env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-12.9/lib64}" \
    NTA_REVISION="${NTA_REVISION:-$(git -C "$root_dir" rev-parse HEAD)}" \
    "$benchmark" \
    --device="vfio:$bdf" \
    --gpu="$gpu" \
    --namespace="$nsid" \
    --queue-depth="$depth" \
    --media-policy="$media_policy" \
    --reference="$reference" \
    "${@:2}"
  "$0" restore
  trap - EXIT
  ;;
restore)
  require_safe_device
  if [[ $(current_driver) == vfio-pci ]]; then
    printf '%s' "$bdf" | sudo tee "$device/driver/unbind" >/dev/null
  fi
  driver=nvmex
  if [[ -r $state ]]; then
    driver=$(<"$state")
  fi
  [[ $driver != none && -d /sys/bus/pci/drivers/$driver ]] || driver=nvme
  printf '%s' "$driver" | sudo tee "$device/driver_override" >/dev/null
  printf '%s' "$bdf" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
  sudo rm -f "$state"
  "$0" status
  ;;
*)
  die "usage: $0 {status|preflight|bind|probe|bind-and-probe|qualify|restore}"
  ;;
esac
