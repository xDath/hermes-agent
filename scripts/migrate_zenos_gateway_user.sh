#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this migration as root." >&2
  exit 1
fi

SOURCE_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_USER="hermes"
SERVICE_GROUP="hermes"
SERVICE_HOME="/var/lib/hermes"
PROFILE_ROOT="${SERVICE_HOME}/.hermes/profiles/zenos"
LEGACY_PROFILE="/root/.hermes/profiles/zenos"
WORKSPACE_ROOT="/root/openclaw-projects"

getent group "${SERVICE_GROUP}" >/dev/null || groupadd --system "${SERVICE_GROUP}"
id -u "${SERVICE_USER}" >/dev/null 2>&1 || useradd --system \
  --gid "${SERVICE_GROUP}" \
  --home-dir "${SERVICE_HOME}" \
  --shell /usr/sbin/nologin \
  "${SERVICE_USER}"
install -d -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" -m 0700 \
  "${SERVICE_HOME}" "${SERVICE_HOME}/.hermes" "${SERVICE_HOME}/.hermes/profiles"

if ! command -v setfacl >/dev/null 2>&1; then
  echo "setfacl is required. Install the operating-system acl package first." >&2
  exit 1
fi

systemctl stop hermes-gateway.service

if [[ -L "${LEGACY_PROFILE}" ]]; then
  resolved="$(readlink -f "${LEGACY_PROFILE}")"
  [[ "${resolved}" == "${PROFILE_ROOT}" ]] || {
    echo "Legacy profile points to unexpected target: ${resolved}" >&2
    exit 1
  }
elif [[ -d "${LEGACY_PROFILE}" && ! -e "${PROFILE_ROOT}" ]]; then
  mv "${LEGACY_PROFILE}" "${PROFILE_ROOT}"
  ln -s "${PROFILE_ROOT}" "${LEGACY_PROFILE}"
elif [[ -d "${LEGACY_PROFILE}" && -d "${PROFILE_ROOT}" ]]; then
  echo "Both legacy and service profiles exist; refusing an ambiguous merge." >&2
  exit 1
elif [[ ! -d "${PROFILE_ROOT}" ]]; then
  echo "No Zenos Hermes profile exists at either supported location." >&2
  exit 1
fi

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${PROFILE_ROOT}"
find "${PROFILE_ROOT}" -xdev -type d -exec chmod u+rwx,go-rwx {} +
find "${PROFILE_ROOT}" -xdev -type f -exec chmod u+rw,go-rwx {} +

# Permit traversal through /root without exposing its directory listing, then
# grant Hermes access only to the workspace tree it is expected to operate on.
setfacl -m "u:${SERVICE_USER}:--x" /root
setfacl -R -m "u:${SERVICE_USER}:rwX" "${WORKSPACE_ROOT}"
find "${WORKSPACE_ROOT}" -xdev -type d \
  -exec setfacl -m "d:u:${SERVICE_USER}:rwX" {} +

runuser -u "${SERVICE_USER}" -- env HOME="${SERVICE_HOME}" \
  git config --global --replace-all safe.directory "${WORKSPACE_ROOT}/*"

install -o root -g root -m 0644 \
  "${SOURCE_ROOT}/deploy/hermes-gateway-zenos.service" \
  /etc/systemd/system/hermes-gateway.service
systemctl daemon-reload
systemctl enable hermes-gateway.service >/dev/null
systemctl restart hermes-gateway.service

for _ in {1..30}; do
  if systemctl is-active --quiet hermes-gateway.service; then
    gateway_user="$(systemctl show hermes-gateway.service -p User --value)"
    [[ "${gateway_user}" == "${SERVICE_USER}" ]] || {
      echo "Gateway is active under unexpected user: ${gateway_user}" >&2
      exit 1
    }
    echo "Hermes gateway is active as ${SERVICE_USER}."
    exit 0
  fi
  sleep 1
done

systemctl --no-pager --full status hermes-gateway.service || true
exit 1
