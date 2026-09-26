#!/usr/bin/env bash
# Install or upgrade the Boltz2 PLT static demo on the shared Ubuntu VM.
#
#   sudo SITE_HOST=boltz2-plt.<ip>.sslip.io bash install.sh /tmp/boltz2-plt.tar.gz /tmp/Caddyfile.snippet
#
# Run by deploy/azure/deploy.sh; safe to re-run. Conventions for the shared VM:
#   * static files only, under /srv/boltz2-plt/www, owned by root, served by Caddy
#   * no new service, user, port or package
#   * the existing Caddyfile is APPENDED to (backed up, validated, rolled back
#     on failure), never replaced; Caddy is only reloaded (graceful)
#   * nothing belonging to any other app is stopped, edited or restarted
#
# Options (env): SITE_HOST (required unless SKIP_CADDY=1), SKIP_CADDY=1 for testing.
set -euo pipefail

TARBALL="${1:?usage: install.sh site.tar.gz Caddyfile.snippet}"
SNIPPET="${2:-}"
ROOT="${ROOT:-/srv/boltz2-plt}"
SKIP_CADDY="${SKIP_CADDY:-0}"
MARK="# ---- Boltz2 PLT (appended by deploy/azure/install.sh) ----"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo)"
[ -f "$TARBALL" ] || die "no such file: $TARBALL"
if [ "$SKIP_CADDY" != 1 ]; then
  [ -n "${SITE_HOST:-}" ] || die "set SITE_HOST (e.g. boltz2-plt.20.25.227.252.sslip.io)"
  [[ "$SITE_HOST" =~ ^[a-z0-9.-]+$ ]] || die "SITE_HOST looks wrong: $SITE_HOST"
  [ -f "$SNIPPET" ] || die "no Caddyfile snippet: $SNIPPET"
  command -v caddy >/dev/null || die "caddy is not installed on this VM"
  [ -f /etc/caddy/Caddyfile ] || die "/etc/caddy/Caddyfile not found"
fi

log "files -> $ROOT/www"
install -d -o root -g root -m 755 "$ROOT"
stage="$ROOT/www.new"
rm -rf "$stage"
install -d -o root -g root -m 755 "$stage"
tar -xzf "$TARBALL" -C "$stage"
[ -f "$stage/index.html" ] || die "tarball has no index.html; nothing was changed"
# Public content: root-owned, world-readable, so Caddy can serve it but cannot change it.
chown -R root:root "$stage"
find "$stage" -type d -exec chmod 755 {} +
find "$stage" -type f -exec chmod 644 {} +
rm -rf "$ROOT/www.old"
[ -d "$ROOT/www" ] && mv "$ROOT/www" "$ROOT/www.old"
mv "$stage" "$ROOT/www"
rm -rf "$ROOT/www.old"
echo "   $(find "$ROOT/www" -type f | wc -l) file(s) in place"

if [ "$SKIP_CADDY" != 1 ]; then
  log "Caddy site block for $SITE_HOST"
  cf=/etc/caddy/Caddyfile
  if grep -qF "$MARK" "$cf"; then
    grep -qF "$SITE_HOST {" "$cf" \
      || die "a Boltz2 PLT block exists with a different host; edit $cf by hand (see deploy/azure/README.md)"
    echo "   block already present; leaving the Caddyfile unchanged"
  else
    backup="$cf.bak-boltz2plt-$(date +%Y%m%d%H%M%S)"
    cp -p "$cf" "$backup"
    { echo; sed "s/__SITE_HOST__/$SITE_HOST/" "$SNIPPET"; } >> "$cf"
    if ! caddy validate --config "$cf" --adapter caddyfile >/dev/null 2>&1; then
      cp -p "$backup" "$cf"
      caddy validate --config "$cf" --adapter caddyfile || true
      die "new Caddyfile failed validation; restored $backup (Caddy was not reloaded)"
    fi
    echo "   appended (backup: $backup)"
  fi
  systemctl reload caddy
  sleep 2
  journalctl -u caddy -n 15 --no-pager || true
fi

log "done"
[ "$SKIP_CADDY" != 1 ] && echo "   site: https://$SITE_HOST  (first request may take ~30 s while Caddy gets a certificate)"
exit 0
