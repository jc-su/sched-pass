#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rules_source=$root_dir/config/udev/99-nta-nvme-p2p.rules
rules_target=/etc/udev/rules.d/99-nta-nvme-p2p.rules
target_user=${1:-${SUDO_USER:-${USER:-}}}

die() {
  printf 'install-nta-device-access: %s\n' "$*" >&2
  exit 1
}

as_root() {
  if [[ $EUID -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

[[ -r $rules_source ]] || die "udev rule is absent: $rules_source"
[[ -n $target_user ]] || die "unable to determine the artifact user"
id "$target_user" >/dev/null 2>&1 || die "unknown user: $target_user"

as_root groupadd --system --force nta
as_root usermod --append --groups nta "$target_user"
as_root install -m 0644 "$rules_source" "$rules_target"
as_root udevadm control --reload-rules
as_root udevadm trigger --subsystem-match=nta_nvme_p2p --action=change
as_root udevadm settle

printf 'nta_device_access=installed user=%s group=nta rule=%s\n' \
  "$target_user" "$rules_target"
if [[ -c /dev/nta_nvme_p2p ]]; then
  ls -l /dev/nta_nvme_p2p
fi
printf '%s\n' \
  'start a new login session before running unprivileged NVMe probe/benchmark commands'
