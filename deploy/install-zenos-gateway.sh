#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this installer as root." >&2
  exit 1
fi

SOURCE_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
install -o root -g root -m 0644 "${SOURCE_ROOT}/deploy/hermes-gateway-zenos.service" /etc/systemd/system/hermes-gateway.service
systemctl daemon-reload
systemctl enable hermes-gateway.service >/dev/null
systemctl restart hermes-gateway.service
systemctl --no-pager --full status hermes-gateway.service
