# Deploying the Boltz2 PLT demo site to the shared Azure VM

This is a runbook meant to be followed by an AI coding agent (or a person) working in
a shell on Rishi's Mac (Apple Silicon, macOS). It deploys the static demo page in
`site/` to the same Azure VM that already hosts **RiskRadar** and **podcast-qna**.

The site is one static HTML file. On the VM it is just files served by the existing
Caddy web server: no new service, user, port, runtime or package.

```
internet ──443──> Caddy ──> /srv/boltz2-plt/www/index.html   (file_server, this project)
                        ├─> 127.0.0.1:3001                    (risk-radar, do not touch)
                        └─> ...                               (podcast-qna, do not touch)
```

## Known facts

| Item | Value |
|---|---|
| VM public IP | `20.25.227.252` |
| SSH login | `azureuser@20.25.227.252` |
| SSH key | `~/.ssh/id_ed25519` (override with `SSH_KEY=...`) |
| OS | Ubuntu 24.04, 2 vCPU, 4 GB RAM |
| Web server | Caddy, config `/etc/caddy/Caddyfile`, automatic HTTPS |
| Hostname for this site | `boltz2-plt.20.25.227.252.sslip.io` |
| Files on the VM | `/srv/boltz2-plt/www/` (root-owned, world-readable) |
| SSH firewall rule | NSG inbound rule named `ssh-home`, allows one home IP |
| Other tenants | `risk-radar.20.25.227.252.sslip.io` (RiskRadar), plus podcast-qna |

`sslip.io` is a public wildcard DNS service, so `boltz2-plt.20.25.227.252.sslip.io`
already resolves to the VM with no DNS setup, and Caddy obtains a TLS certificate
for it on the first request.

## Ground rules (read before running anything)

1. **Do not stop, restart, edit or redeploy anything that belongs to RiskRadar or
   podcast-qna.** Caddy may only be *reloaded* (graceful), never restarted.
2. **Never replace `/etc/caddy/Caddyfile`.** `install.sh` only appends one marked
   block, backs the file up first, validates it, and restores the backup on failure.
   Do not hand-edit other apps' blocks.
3. **The only Azure resource you may change is the `ssh-home` NSG rule**, and only its
   source IP, and only if SSH is blocked because the home IP changed.
4. Deploy only committed code. `deploy.sh` refuses to run with uncommitted changes
   under `site/` or `deploy/`.
5. If anything unexpected happens (another app's health changes, Caddy reload errors,
   disk or memory is tight), stop and report to the user instead of improvising.

## Step 0. Tools on the Mac

```bash
git --version && ssh -V && curl --version | head -1
az version >/dev/null 2>&1 || echo "Azure CLI missing: brew install azure-cli"   # only needed for step 2
gh --version  >/dev/null 2>&1 || echo "GitHub CLI missing: brew install gh"         # only needed for step 6
```

## Step 1. Get the code

```bash
cd ~/code 2>/dev/null || cd ~
[ -d boltz2-plt ] || git clone https://github.com/rishimj/boltz2-plt.git
cd boltz2-plt
git fetch origin
git checkout main && git pull --ff-only
# If the site has not been merged to main yet, deploy the feature branch instead:
#   git checkout claude/admiring-sagan-onhppg && git pull --ff-only
ls site/index.html deploy/azure/deploy.sh deploy/azure/install.sh deploy/azure/Caddyfile.snippet
git status --short    # should print nothing for site/ and deploy/
```

## Step 2. Make sure SSH works

```bash
ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=10 -o BatchMode=yes azureuser@20.25.227.252 'echo ssh-ok'
```

If this prints `ssh-ok`, go to step 3.

If it **times out**, the home IP has most likely changed and the NSG blocks it. Fix only
the `ssh-home` rule:

```bash
MYIP=$(curl -s https://api.ipify.org); echo "current IP: $MYIP"
az account show -o table || az login

# Find the NSG that has the ssh-home rule (resource group and NSG name).
az network nsg list --query "[].{nsg:name, rg:resourceGroup}" -o table
# For the NSG attached to the VM, confirm the rule and its current source:
az network nsg rule show -g <RG> --nsg-name <NSG> -n ssh-home \
  --query "{src:sourceAddressPrefix, port:destinationPortRange, access:access}" -o table

az network nsg rule update -g <RG> --nsg-name <NSG> -n ssh-home \
  --source-address-prefixes "$MYIP/32"
```

Tell the user the old IP loses SSH access. Retry the SSH check (rules apply within a
minute). If it fails with `Permission denied (publickey)`, it is a key problem, not the
firewall: stop and ask the user.

## Step 3. Preflight on the VM

Record the other apps' state **before** deploying, so you can prove nothing changed.

```bash
VM=azureuser@20.25.227.252
ssh $VM 'free -h; df -h /srv | tail -1; systemctl is-active caddy; caddy version'
ssh $VM 'grep -n "^# ----\|sslip.io\|^[a-z0-9.-]\+ {" /etc/caddy/Caddyfile'
```

From the Caddyfile listing, note every public hostname (RiskRadar, podcast-qna, and a
Boltz2 PLT block if a previous deploy already added one). Then record their status:

```bash
for h in risk-radar.20.25.227.252.sslip.io <podcast-qna-host>; do
  printf '%-45s %s\n' "$h" "$(curl -s -o /dev/null -w '%{http_code}' https://$h/)"
done
curl -s https://risk-radar.20.25.227.252.sslip.io/healthz; echo
```

Expected: Caddy `active`, a few hundred MB of disk free (the site needs about 100 KB),
other apps answering 200 (or whatever they answered before; just record it).

## Step 4. Deploy

From the repo root on the Mac:

```bash
VM=azureuser@20.25.227.252 SITE_HOST=boltz2-plt.20.25.227.252.sslip.io deploy/azure/deploy.sh
```

What happens:

1. `deploy.sh` packs the committed `site/` folder with `git archive` and copies it,
   `install.sh` and `Caddyfile.snippet` to `/tmp` on the VM.
2. `install.sh` (run with sudo on the VM) unpacks into `/srv/boltz2-plt/www.new`, sets
   root ownership and `755/644` permissions, and swaps it into `/srv/boltz2-plt/www`.
3. On the first deploy only, it appends the `# ---- Boltz2 PLT ...` block to the
   Caddyfile, runs `caddy validate`, restores the backup if validation fails, then
   runs `systemctl reload caddy`. On later deploys the block is already there and the
   Caddyfile is not touched; only the files change.
4. `deploy.sh` polls `https://boltz2-plt.20.25.227.252.sslip.io/` until the page title
   appears (the first request can take about 30 s while Caddy gets a certificate).

Success ends with `live: https://boltz2-plt.20.25.227.252.sslip.io`.

## Step 5. Verify

```bash
H=boltz2-plt.20.25.227.252.sslip.io
curl -sI https://$H/ | grep -iE "^HTTP|content-type|cache-control|strict-transport"
curl -s  https://$H/ | grep -o "<title>.*</title>"          # <title>Boltz2 PLT Explorer</title>
# Served bytes must match the committed file exactly:
[ "$(curl -s https://$H/ | shasum -a 256 | cut -d' ' -f1)" = "$(git show HEAD:site/index.html | shasum -a 256 | cut -d' ' -f1)" ] && echo "matches HEAD"
```

Re-run the step 3 status loop and confirm RiskRadar and podcast-qna give the same codes
as before. Also check Caddy logged no errors:

```bash
ssh azureuser@20.25.227.252 'sudo journalctl -u caddy -n 30 --no-pager | grep -i error || echo "no caddy errors"'
```

Finally, open the URL in a browser and confirm the 3D protein in the hero rotates
and the "Run transcoder" button works. Fonts load from Google Fonts, which is expected.

## Step 6. Link it (so recruiters find it)

Only after step 5 passes:

```bash
URL=https://boltz2-plt.20.25.227.252.sslip.io
# README links currently point at GitHub Pages; switch them to the live Azure URL.
grep -n "rishimj.github.io/boltz2-plt" README.md
sed -i '' "s#https://rishimj.github.io/boltz2-plt/#$URL/#g" README.md     # macOS sed
git add README.md && git commit -m "Point live demo links at the Azure deployment" && git push

gh repo edit rishimj/boltz2-plt --homepage "$URL" \
  --description "Per-Layer Transcoders: sparse autoencoders for interpreting Boltz2 protein structure prediction"
```

Commit README changes on whatever branch the user is working on (ask if unsure); do not
push directly to `main` unless the user says to. The repo also has a GitHub Pages workflow
(`.github/workflows/pages.yml`). It is harmless to keep as a backup mirror; ask the user
before deleting it.

## Redeploy later

After changing `site/index.html` and committing:

```bash
VM=azureuser@20.25.227.252 SITE_HOST=boltz2-plt.20.25.227.252.sslip.io deploy/azure/deploy.sh
```

To roll back, check out an earlier commit (`git checkout <sha>`) and run the same command,
then return to your branch.

## Uninstall

```bash
ssh azureuser@20.25.227.252
sudo cp -p /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak-boltz2plt-remove
# delete ONLY the lines from "# ---- Boltz2 PLT" through "# ---- end Boltz2 PLT ----":
sudo sed -i '/^# ---- Boltz2 PLT (appended/,/^# ---- end Boltz2 PLT ----/d' /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy
sudo rm -rf /srv/boltz2-plt
```

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| SSH times out | Home IP changed. Step 2. |
| `Permission denied (publickey)` | Wrong key. Try `SSH_KEY=~/.ssh/<other key>`; otherwise ask the user. |
| `deploy.sh`: uncommitted changes | Commit or stash changes under `site/` or `deploy/`. |
| `install.sh`: block exists with a different host | A previous deploy used another hostname. Ask the user which host to keep, then edit only the Boltz2 PLT block by hand (backup, `caddy validate`, `systemctl reload caddy`). |
| `new Caddyfile failed validation` | Nothing was changed; the backup was restored. Show the `caddy validate` output to the user. |
| Site not answering after 2 minutes | `ssh $VM sudo journalctl -u caddy -n 50`. Certificate errors usually mean ports 80/443 are closed in the NSG, but they are already open for RiskRadar, so check the hostname for typos. |
| 403 or 404 on `/` | Permissions or missing file: `ssh $VM 'ls -la /srv/boltz2-plt/www'` should show `index.html` as `-rw-r--r-- root root`. Re-run the deploy. |
| Old page still shows | The HTML is sent with `Cache-Control: no-cache`; hard-refresh the browser (Cmd+Shift+R). |
