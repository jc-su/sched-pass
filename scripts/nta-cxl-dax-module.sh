#!/usr/bin/env bash
set -euo pipefail

# Load only the kernel plumbing required to consume a platform-provisioned
# CXL Type-3 devdax endpoint.  This script never creates a region, changes a
# decoder, formats media, or changes a dax namespace.

die() {
  printf 'nta-cxl-dax-module: %s\n' "$*" >&2
  exit 1
}

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
  command -v cxl >/dev/null 2>&1 || die "cxl utility is not installed"
  command -v daxctl >/dev/null 2>&1 || die "daxctl utility is not installed"
  printf 'cxl_memdevs=\n'
  cxl list -M -i 2>&1 || true
  printf 'cxl_regions=\n'
  cxl list -R 2>&1 || true
  printf 'devdax=\n'
  daxctl list -u 2>&1 || true
}

case ${1:-status} in
load)
  for module in cxl_mem dax_cxl device_dax; do
    as_root modprobe "$module"
  done
  printf 'kernel=%s %s %s %s\n' "$(uname -r)" \
    "$(module_state cxl_mem)" "$(module_state dax_cxl)" \
    "$(module_state device_dax)"
  inventory
  ;;
status)
  printf 'kernel=%s %s %s %s\n' "$(uname -r)" \
    "$(module_state cxl_mem)" "$(module_state dax_cxl)" \
    "$(module_state device_dax)"
  inventory
  ;;
*)
  die "usage: $0 {load|status}"
  ;;
esac
