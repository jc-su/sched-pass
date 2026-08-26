#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
module_dir=$root_dir/kernel/nta_nvme_p2p
kernel_release=$(uname -r)

die() {
  printf 'nta-nvme-p2p-module: %s\n' "$*" >&2
  exit 1
}

as_root() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

nvidia_version() {
  [[ -r /sys/module/nvidia/version ]] || die "the NVIDIA kernel driver is not loaded"
  tr -d '\n' </sys/module/nvidia/version
}

build_module() {
  local version include_dir symbols
  version=$(nvidia_version)
  include_dir=${NTA_NVIDIA_P2P_INCLUDE:-/usr/src/nvidia-$version/nvidia-peermem}
  symbols=${NTA_NVIDIA_MODULE_SYMVERS:-\
/var/lib/dkms/nvidia/$version/$kernel_release/x86_64/module/Module.symvers}
  [[ -d /lib/modules/$kernel_release/build ]] ||
    die "kernel headers for $kernel_release are absent"
  [[ -r $include_dir/nv-p2p.h ]] || die "NVIDIA peer-memory headers are absent"
  [[ -r $symbols ]] || die "NVIDIA Module.symvers for $kernel_release is absent"
  make -C "/lib/modules/$kernel_release/build" M="$module_dir" \
    NVIDIA_P2P_INCLUDE="$include_dir" KBUILD_EXTRA_SYMBOLS="$symbols" modules
}

case ${1:-status} in
build)
  build_module
  ;;
load)
  build_module
  if [[ -d /sys/module/nta_nvme_p2p ]]; then
    loaded_srcversion=$(</sys/module/nta_nvme_p2p/srcversion)
    built_srcversion=$(modinfo -F srcversion "$module_dir/nta_nvme_p2p.ko")
    [[ -n $loaded_srcversion && $loaded_srcversion == "$built_srcversion" ]] ||
      die "a different nta_nvme_p2p build is loaded; unload it before loading this revision"
  else
    as_root insmod "$module_dir/nta_nvme_p2p.ko"
  fi
  [[ -c /dev/nta_nvme_p2p ]] || die "module loaded without a device node"
  printf 'module=nta_nvme_p2p state=loaded device=/dev/nta_nvme_p2p kernel=%s\n' \
    "$kernel_release"
  ;;
unload)
  if [[ -d /sys/module/nta_nvme_p2p ]]; then
    as_root rmmod nta_nvme_p2p
  fi
  printf 'module=nta_nvme_p2p state=unloaded\n'
  ;;
status)
  if [[ -d /sys/module/nta_nvme_p2p ]]; then
    printf 'module=nta_nvme_p2p state=loaded device=%s kernel=%s\n' \
      "$(test -c /dev/nta_nvme_p2p && printf present || printf absent)" \
      "$kernel_release"
  else
    printf 'module=nta_nvme_p2p state=unloaded kernel=%s\n' "$kernel_release"
  fi
  ;;
*)
  die "usage: $0 {build|load|unload|status}"
  ;;
esac
