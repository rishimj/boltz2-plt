#!/usr/bin/env bash
# Deploy the committed site/ folder to the Azure VM. Run from your laptop at the repo root:
#
#   VM=azureuser@<vm-ip> SITE_HOST=boltz2-plt.<vm-ip>.sslip.io deploy/azure/deploy.sh
#
# Ships only tracked files under site/ (git archive), so nothing untracked can
# leak onto the server. Works on macOS and Linux.
set -euo pipefail
: "${VM:?set VM=user@host}"
: "${SITE_HOST:?set SITE_HOST=boltz2-plt.<ip>.sslip.io}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

cd "$(git rev-parse --show-toplevel)"
if ! git diff --quiet HEAD -- site deploy; then
  echo "!! uncommitted changes under site/ or deploy/; deploy ships HEAD only. Commit or stash first." >&2
  exit 1
fi
git cat-file -e HEAD:site/index.html 2>/dev/null || { echo "!! HEAD has no site/index.html" >&2; exit 1; }

tarball=$(mktemp -t boltz2-plt.XXXXXX)
git archive --format=tar.gz -o "$tarball" HEAD:site
echo "==> shipping site/ from $(git rev-parse --short HEAD) ($(du -h "$tarball" | cut -f1))"

scp -i "$SSH_KEY" -q "$tarball" "$VM:/tmp/boltz2-plt.tar.gz"
scp -i "$SSH_KEY" -q deploy/azure/install.sh "$VM:/tmp/boltz2-plt-install.sh"
scp -i "$SSH_KEY" -q deploy/azure/Caddyfile.snippet "$VM:/tmp/boltz2-plt-Caddyfile.snippet"
ssh -i "$SSH_KEY" "$VM" "sudo SITE_HOST='$SITE_HOST' bash /tmp/boltz2-plt-install.sh /tmp/boltz2-plt.tar.gz /tmp/boltz2-plt-Caddyfile.snippet \
  && rm -f /tmp/boltz2-plt.tar.gz /tmp/boltz2-plt-install.sh /tmp/boltz2-plt-Caddyfile.snippet"
rm -f "$tarball"

echo "==> checking https://$SITE_HOST"
for i in $(seq 1 30); do
  # Capture first: with pipefail, `curl | grep -q` fails when grep exits early (SIGPIPE).
  page=$(curl -sf "https://$SITE_HOST/" || true)
  if grep -qF "<title>Boltz2 PLT Explorer</title>" <<<"$page"; then
    echo "   live: https://$SITE_HOST"; exit 0
  fi
  sleep 4
done
echo "!! https://$SITE_HOST not answering yet; check: ssh $VM sudo journalctl -u caddy -n 50" >&2
exit 1
