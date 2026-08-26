#!/usr/bin/env bash
set -euo pipefail

# Load only the kernel plumbing required to consume a platform-provisioned
# CXL Type-3 devdax endpoint.  This script never creates a region, changes a
# decoder, formats media, or changes a dax namespace.

die() {
  printf 'nta-cxl-dax-module: %s\n' "$*" >&2
  exit 1
}

# Keep the complete, read-only CXL-to-devdax plumbing in one place.  Loading a
# module does not create a CXL endpoint; firmware and a Type-3 device must
# still enumerate the topology.  cxl_pci is included explicitly because a
# PCIe CXL.mem endpoint will otherwise remain invisible even when cxl_mem is
# available as a module.
CXL_MODULES=(
  cxl_core
  cxl_port
  cxl_pci
  cxl_mem
  cxl_acpi
  cxl_pmem
  dax_cxl
  device_dax
)

as_root() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

module_state() {
  local module=$1
  if [[ -d /sys/module/$module ]]; then
    printf '%s=loaded' "$module"
  else
    printf '%s=unloaded' "$module"
  fi
}

sysfs_count() {
  local root=$1
  local pattern=$2
  if [[ ! -d $root ]]; then
    printf '0'
    return
  fi
  find "$root" -maxdepth 1 -mindepth 1 -type d -name "$pattern" -printf '.' 2>/dev/null | wc -c
}

pci_device_count() {
  local expected_vendor=$1
  local expected_device=$2
  local device count=0 vendor product
  for device in /sys/bus/pci/devices/*; do
    [[ -d $device ]] || continue
    vendor=$(cat "$device/vendor" 2>/dev/null || true)
    product=$(cat "$device/device" 2>/dev/null || true)
    if [[ $vendor == "$expected_vendor" && $product == "$expected_device" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s' "$count"
}

topology_state() {
  local memdevs regions dax_nodes type2_endpoints
  memdevs=$(sysfs_count /sys/bus/cxl/devices 'mem[0-9]*')
  regions=$(sysfs_count /sys/bus/cxl/devices 'region[0-9]*')
  dax_nodes=$(sysfs_count /sys/class/dax 'dax[0-9]*.[0-9]*')
  type2_endpoints=$(pci_device_count 0x8086 0x0ddb)
  if (( dax_nodes > 0 )); then
    printf 'devdax_ready'
  elif (( regions > 0 )); then
    printf 'region_without_devdax'
  elif (( memdevs > 0 )); then
    printf 'type3_memdev_without_region'
  elif (( type2_endpoints > 0 )); then
    printf 'type2_pci_endpoint_without_memdev'
  elif [[ -d /sys/bus/cxl/devices/root0 ]]; then
    printf 'root_decoders_only'
  else
    printf 'no_cxl_topology'
  fi
}

inventory() {
  printf 'cxl_memdevs=\n'
  if command -v cxl >/dev/null 2>&1; then
    cxl list -M -i 2>&1 || true
  else
    printf 'unavailable (cxl utility is not installed)\n'
  fi
  printf 'cxl_regions=\n'
  if command -v cxl >/dev/null 2>&1; then
    cxl list -R 2>&1 || true
  else
    printf 'unavailable (cxl utility is not installed)\n'
  fi
  printf 'devdax=\n'
  if command -v daxctl >/dev/null 2>&1; then
    daxctl list -u 2>&1 || true
  else
    printf 'unavailable (daxctl utility is not installed)\n'
  fi
}

print_module_state() {
  local type2_endpoints
  type2_endpoints=$(pci_device_count 0x8086 0x0ddb)
  printf 'kernel=%s' "$(uname -r)"
  for module in "${CXL_MODULES[@]}"; do
    printf ' %s' "$(module_state "$module")"
  done
  printf ' topology=%s type2_endpoints=%s' "$(topology_state)" "$type2_endpoints"
  if [[ -e /sys/firmware/acpi/tables/CEDT ]]; then
    printf ' cedt=present'
  else
    printf ' cedt=absent'
  fi
  printf '\n'
}

case ${1:-status} in
load)
  for module in "${CXL_MODULES[@]}"; do
    as_root modprobe "$module"
  done
  print_module_state
  inventory
  ;;
status)
  print_module_state
  inventory
  ;;
*)
  die "usage: $0 {load|status}"
  ;;
esac
