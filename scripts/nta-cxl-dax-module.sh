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
    sudo "$@"
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
  printf 'kernel=%s' "$(uname -r)"
  for module in "${CXL_MODULES[@]}"; do
    printf ' %s' "$(module_state "$module")"
  done
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
