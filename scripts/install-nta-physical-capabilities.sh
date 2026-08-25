#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${1:-$root_dir/build}

die() {
  printf 'install-nta-physical-capabilities: %s\n' "$*" >&2
  exit 1
}

[[ -d $build_dir ]] || die "build directory does not exist: $build_dir"
command -v setcap >/dev/null 2>&1 || die "setcap is required (install libcap2-bin)"

targets=(
  "$build_dir/nta-vfio-nvme-probe"
  "$build_dir/nta-nvme-bench"
  "$build_dir/nta-paged-attention"
)
for target in "${targets[@]}"; do
  [[ -f $target && -x $target ]] ||
    die "physical executable is absent or not executable: $target"
done

for target in "${targets[@]}"; do
  if [[ $EUID -eq 0 ]]; then
    setcap cap_sys_admin+ep "$target"
  else
    sudo setcap cap_sys_admin+ep "$target"
  fi
  printf 'physical_capability=cap_sys_admin executable=%s\n' "$target"
done

cat <<'EOF'
The capability is intentionally limited to the three physical NVMe tools.
It is required by the NVIDIA driver for cuMemHostRegister of the VFIO BAR
doorbell. It does not grant raw NVMe namespace access, and it is not used by
the framework adapters or resident serving path. Re-run this setup after a
rebuild that replaces these executables; remove it with setcap -r when the
physical tier is no longer in use.
EOF
