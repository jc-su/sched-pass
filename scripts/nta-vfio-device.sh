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
dma_target=${NTA_NVME_DMA_TARGET:-hbm-peer}
probe=${NTA_VFIO_PROBE:-$root_dir/build/nta-vfio-nvme-probe}
benchmark=${NTA_NVME_BENCHMARK:-$root_dir/build/nta-nvme-bench}

die() {
  printf 'nta-vfio-device: %s\n' "$*" >&2
  exit 1
}

cuda_library_path() {
  if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
    printf '%s\n' "$LD_LIBRARY_PATH"
    return
  fi
  local python_bin=${NTA_PYTHON:-python3}
  local requested=${NTA_CUDA_ROOT:-${NTA_CUDA_PATH:-${CUDAToolkit_ROOT:-}}}
  local cuda_home
  if [[ -n $requested ]]; then
    cuda_home=$(
      "$python_bin" "$root_dir/tools/jit/cuda_toolkit.py" \
        --cuda-path "$requested" --print-root
    )
  else
    cuda_home=$(
      "$python_bin" "$root_dir/tools/jit/cuda_toolkit.py" --print-root
    )
  fi
  printf '%s/lib64\n' "$cuda_home"
}

has_cuda_admin_capability() {
  local executable=$1 capabilities
  command -v getcap >/dev/null 2>&1 || return 1
  [[ -x $executable ]] || return 1
  capabilities=$(getcap "$executable" 2>/dev/null || true)
  [[ $capabilities == *cap_sys_admin=ep* ||
    $capabilities == *cap_sys_admin+ep* ]]
}

resolve_privilege_mode() {
  local requested=$1 executable=$2
  case $requested in
  0|1)
    printf '%s\n' "$requested"
    ;;
  auto)
    if has_cuda_admin_capability "$executable"; then
      printf '0\n'
    else
      printf '1\n'
    fi
    ;;
  *)
    die "privilege mode must be auto, 0, or 1"
    ;;
  esac
}

probe_as_root=$(resolve_privilege_mode "${NTA_VFIO_PROBE_SUDO:-auto}" "$probe")
benchmark_as_root=$(resolve_privilege_mode \
  "${NTA_VFIO_BENCHMARK_SUDO:-auto}" "$benchmark")
state=/run/nta-vfio-${bdf}.driver
partition_state=/run/nta-vfio-${bdf}.partitions

[[ $bdf =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]] ||
  die "NTA_NVME_BDF must use DDDD:BB:SS.F syntax"

current_driver() {
  if [[ -L $device/driver ]]; then
    basename "$(readlink "$device/driver")"
  else
    printf 'none\n'
  fi
}

wait_for_driver() {
  local expected=$1
  local _
  for _ in {1..100}; do
    [[ $(current_driver) == "$expected" ]] && return 0
    sleep 0.1
  done
  die "$bdf did not bind back to $expected"
}

wait_for_nvme_namespace() {
  local _ controller block
  command -v udevadm >/dev/null 2>&1 && sudo udevadm settle --timeout=10 || true
  for _ in {1..100}; do
    for controller in "$device"/nvme/nvme*; do
      [[ -d $controller ]] || continue
      [[ $(<"$controller/state") == live ]] || continue
      for block in "$controller"/nvme*n*; do
        [[ -e $block ]] || continue
        [[ $(<"$block/nsid") == "$nsid" ]] || continue
        [[ -b /dev/$(basename "$block") ]] || continue
        printf 'restored_namespace=/dev/%s\n' "$(basename "$block")"
        return 0
      done
    done
    sleep 0.1
  done
  die "$bdf returned to the nvme driver but no live namespace appeared"
}

wait_for_expected_partitions() {
  [[ -r $partition_state ]] || return 0
  local partition _
  while IFS= read -r partition; do
    [[ -n $partition ]] || continue
    for _ in {1..100}; do
      [[ -b /dev/$partition ]] && break
      sleep 0.1
    done
    [[ -b /dev/$partition ]] ||
      die \
        "$bdf returned to nvme but expected partition /dev/$partition did not reappear"
  done < "$partition_state"
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

check_block_device() {
  local sysfs_block=$1
  local block_device
  block_device=/dev/$(basename "$sysfs_block")
  [[ -b $block_device ]] || die "$block_device has no block device node"
  if findmnt -rn -S "$block_device" >/dev/null; then
    die "$block_device is mounted"
  fi
  if [[ -n $(ls -A "/sys/class/block/$(basename "$sysfs_block")/holders" 2>/dev/null) ]]; then
    die "$block_device has kernel block holders"
  fi
  if sudo fuser "$block_device" >/dev/null 2>&1; then
    die "$block_device is open by another process"
  fi
}

require_safe_device() {
  [[ -d $device ]] || die "PCI device $bdf does not exist"
  local namespace_found=0
  local block
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    namespace_found=1
    local class_block
    class_block=/sys/class/block/$(basename "$block")
    local hidden
    hidden=$(cat "$class_block/hidden" 2>/dev/null || true)
    if [[ $hidden == 1 ]]; then
      die "$(basename "$block") is a hidden multipath namespace"
    fi

    check_block_device "$block"
    local partition
    for partition in "$block"/"$(basename "$block")"p*; do
      [[ -e $partition ]] || continue
      check_block_device "$partition"
    done
  done
  if [[ $namespace_found == 0 && $(current_driver) != vfio-pci ]]; then
    die "NVMe controller has no namespace"
  fi
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
  [[ $dma_target == hbm-peer || $dma_target == host-mapped ]] ||
    die "NTA_NVME_DMA_TARGET must be hbm-peer or host-mapped"
}

require_rebind_confirmation() {
  [[ ${NTA_ALLOW_DEVICE_REBIND:-0} == 1 ]] ||
    die "device rebind is destructive; set NTA_ALLOW_DEVICE_REBIND=1 after reviewing preflight"
}

require_media_policy() {
  [[ $media_policy == hardware-write-protect ]] || return 0
  command -v nvme >/dev/null 2>&1 ||
    die "hardware-write-protect requires nvme-cli for a read-only capability preflight"

  local block controller controller_output nwpc nsid
  local checked=0
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    controller=$(basename "$(dirname "$block")")
    nsid=$(<"$block/nsid")
    controller_output=$(sudo nvme id-ctrl "/dev/$controller" 2>&1) ||
      die "read-only NVMe Identify Controller failed for /dev/$controller"
    nwpc=$(awk '$1 == "nwpc" { print $3; exit }' <<<"$controller_output")
    [[ -n $nwpc ]] ||
      die "NVMe Identify Controller did not report NWPC for /dev/$controller"
    if (( nwpc == 0 )); then
      die "/dev/$controller namespace $nsid lacks NVMe Namespace Write Protection"
    fi
    sudo nvme get-feature "/dev/$controller" -f 0x84 -n "$nsid" -H >/dev/null 2>&1 ||
      die "read-only Namespace Write Protection Get Feature failed for /dev/$controller namespace $nsid"
    checked=$((checked + 1))
  done
  [[ $checked -gt 0 ]] || die "no active namespace available for media-policy preflight"
}

require_hbm_peer() {
  [[ $dma_target == hbm-peer ]] || return 0
  [[ -d /sys/module/nta_nvme_p2p ]] ||
    die "nta_nvme_p2p is not loaded; run scripts/nta-nvme-p2p-module.sh load"
  [[ -c /dev/nta_nvme_p2p ]] ||
    die "/dev/nta_nvme_p2p is unavailable"
  if [[ $probe_as_root == 1 ]]; then
    if ! sudo test -r /dev/nta_nvme_p2p ||
      ! sudo test -w /dev/nta_nvme_p2p; then
      die "root cannot access /dev/nta_nvme_p2p"
    fi
  else
    [[ -r /dev/nta_nvme_p2p && -w /dev/nta_nvme_p2p ]] ||
      die "current user cannot access /dev/nta_nvme_p2p"
  fi
}

capture_reference() {
  local block
  for block in "$device"/nvme/nvme*/nvme*n*; do
    [[ -e $block ]] || continue
    [[ $(<"$block/nsid") == "$nsid" ]] || continue
    sudo dd if="/dev/$(basename "$block")" bs=4096 \
      count="$((reference_bytes / 4096))" \
      iflag=direct status=none | write_reference bs=4096
    [[ -f $reference && $(stat -c '%s' "$reference") == "$reference_bytes" ]] ||
      die "reference capture has an unexpected size"
    return
  done
  [[ -f $reference ]] ||
    die "no namespace block device exists and reference file is absent"
}

write_reference() {
  local owner_uid
  if [[ $EUID -eq 0 && -e $reference ]]; then
    owner_uid=$(stat -c '%u' "$reference")
    if [[ $owner_uid != 0 ]]; then
      # fs.protected_regular can reject root opening another user's file in a
      # sticky directory such as /tmp.  Preserve the artifact owner instead of
      # weakening that protection or replacing the existing file.
      sudo -u "#$owner_uid" dd of="$reference" status=none "$@"
      return
    fi
  fi
  dd of="$reference" status=none "$@"
}

snapshot_partitions() {
  local block partition
  local partitions
  partitions=$(
    for block in "$device"/nvme/nvme*/nvme*n*; do
      [[ -e $block ]] || continue
      for partition in "$block"p*; do
        [[ -e $partition ]] || continue
        printf '%s\n' "$(basename "$partition")"
      done
    done
  )
  if [[ -n $partitions ]]; then
    printf '%s\n' "$partitions" | sudo tee "$partition_state" >/dev/null
    sudo chmod 0644 "$partition_state"
  else
    sudo rm -f "$partition_state"
  fi
}

bind_vfio() {
  local vmem_users
  vmem_users=$(lsmod | awk '$1 == "vmem_sw" { print $3 }')
  if [[ -n $vmem_users && $vmem_users -ne 0 ]]; then
    die "vmem_sw has active references; stop its users before binding"
  fi
  require_safe_device
  require_containment
  require_media_policy
  require_hbm_peer
  snapshot_partitions
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
  require_media_policy
  require_hbm_peer
  printf 'VFIO preflight passed: bdf=%s nsid=%s depth=%s gpu=%s media_policy=%s dma_target=%s\n' \
    "$bdf" "$nsid" "$depth" "$gpu" "$media_policy" "$dma_target"
  ;;
bind)
  require_rebind_confirmation
  bind_vfio
  "$0" status
  ;;
probe)
  [[ $(current_driver) == vfio-pci ]] || die "$bdf is not bound to vfio-pci"
  [[ -x $probe ]] || die "probe executable is absent; build nta-vfio-nvme-probe"
  if [[ $probe_as_root == 1 ]]; then
    sudo env LD_LIBRARY_PATH="$(cuda_library_path)" \
      "$probe" "vfio:$bdf" "$gpu" "$nsid" "$depth" "$media_policy" \
      "$dma_target"
  else
    "$probe" "vfio:$bdf" "$gpu" "$nsid" "$depth" "$media_policy" \
      "$dma_target"
  fi
  ;;
bind-and-probe)
  require_rebind_confirmation
  restore_on_exit() {
    "$0" restore >/dev/null 2>&1 || true
  }
  trap restore_on_exit EXIT
  bind_vfio
  if ! "$0" probe; then
    printf 'VFIO probe failed; restoring the previous PCI driver\n' >&2
    "$0" restore
    exit 1
  fi
  "$0" restore
  trap - EXIT
  ;;
qualify)
  require_rebind_confirmation
  keep_vfio=${NTA_NVME_KEEP_VFIO:-0}
  [[ $keep_vfio == 0 || $keep_vfio == 1 ]] ||
    die "NTA_NVME_KEEP_VFIO must be 0 or 1"
  restore_on_exit() {
    "$0" restore >/dev/null 2>&1 || true
  }
  trap restore_on_exit EXIT
  if [[ $(current_driver) == vfio-pci ]]; then
    # Reusing an explicitly owned controller avoids an unnecessary PCI
    # unbind/rebind between qualification runs. The containment and peer-DMA
    # checks remain mandatory for the persistent-session path.
    require_containment
    require_hbm_peer
    [[ -r $state ]] ||
      die "VFIO controller has no ownership state; use bind first"
  else
    bind_vfio
  fi
  [[ -x $benchmark ]] || die "benchmark executable is absent; build nta-nvme-bench"
  benchmark_command=(
    env
    "LD_LIBRARY_PATH=$(cuda_library_path)"
    "NTA_REVISION=${NTA_REVISION:-$(git -C "$root_dir" rev-parse HEAD)}"
    "$benchmark"
    "--device=vfio:$bdf"
    "--gpu=$gpu"
    "--namespace=$nsid"
    "--queue-depth=$depth"
    "--media-policy=$media_policy"
    "--dma-target=$dma_target"
    "--reference=$reference"
    "${@:2}"
  )
  if [[ $benchmark_as_root == 1 ]]; then
    sudo "${benchmark_command[@]}"
  else
    "${benchmark_command[@]}"
  fi
  if [[ $keep_vfio == 1 ]]; then
    trap - EXIT
    printf 'persistent_vfio=1 bdf=%s driver=vfio-pci\n' "$bdf"
  else
    "$0" restore
    trap - EXIT
  fi
  ;;
restore)
  require_safe_device
  if [[ $(current_driver) == vfio-pci ]]; then
    printf '%s' "$bdf" | sudo tee "$device/driver/unbind" >/dev/null
  fi
  driver=nvme
  if [[ -r $state ]]; then
    driver=$(<"$state")
  fi
  [[ $driver != none && -d /sys/bus/pci/drivers/$driver ]] || driver=nvme
  printf '%s' "$driver" | sudo tee "$device/driver_override" >/dev/null
  printf '%s' "$bdf" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
  wait_for_driver "$driver"
  # driver_override is only a transactional bind aid. Leaving it set makes
  # later hotplug and recovery depend on stale experiment state.
  printf '\n' | sudo tee "$device/driver_override" >/dev/null
  [[ $driver != nvme ]] || wait_for_nvme_namespace
  [[ $driver != nvme ]] || wait_for_expected_partitions
  sudo rm -f "$state"
  sudo rm -f "$partition_state"
  "$0" status
  ;;
*)
  die "usage: $0 {status|preflight|bind|probe|bind-and-probe|qualify|restore}"
  ;;
esac
