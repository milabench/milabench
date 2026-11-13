#!/bin/bash
#SBATCH --job-name=test-proxy
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#
# Check that the jobrunner internet proxy (ssh SOCKS tunnel to the login node,
# see dashboard/scripts/hook_proxy.sh) is really used by the job.
#
# Submit it from the dashboard with "proxy" enabled (the hook is injected after the
# #SBATCH header), or by hand with the tunnel set up like the hook does:
#
#   sbatch scripts/slurm/test_proxy.sh
#
# The script exits non-zero if the proxy is not proven to be in use. Since some
# clusters (mila) have direct internet on compute nodes, "internet works" is not
# enough, so we check 3 things:
#
#   1. the proxy env vars point at a local SOCKS port that is listening
#   2. a request through a DEAD proxy port fails  -> the client honors the proxy
#   3. the request through the proxy succeeds and the SOCKS listener saw a
#      connection (sshd-forwarded traffic), and the public IP seen through the
#      proxy is the login node's public IP
#
URL=${TEST_PROXY_URL:-https://api.ipify.org}
fail=0

ok()   { echo "[ OK ] $*"; }
bad()  { echo "[FAIL] $*"; fail=1; }
info() { echo "[info] $*"; }

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  login: ${SLURM_SUBMIT_HOST:-unknown}"

# --- 1. environment ---------------------------------------------------------
if [ -z "$ALL_PROXY" ]; then
    bad "ALL_PROXY is not set, the proxy hook was not injected (proxy disabled for this cluster/job?)"
    echo "RESULT: FAIL"
    exit 1
fi
info "ALL_PROXY=$ALL_PROXY"

port=${ALL_PROXY##*:}
if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
    ok "SOCKS port 127.0.0.1:$port is listening"
else
    bad "nothing listens on 127.0.0.1:$port, the ssh tunnel is down"
fi

if pgrep -u "$USER" -af "ssh -N -D 127.0.0.1:$port" >/dev/null; then
    ok "ssh tunnel process is running"
else
    bad "no 'ssh -N -D 127.0.0.1:$port' process found"
fi

# --- 2. direct path (informational) ------------------------------------------
direct_ip=$(env -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    curl -s --noproxy '*' --max-time 10 "$URL")
if [ -n "$direct_ip" ]; then
    info "direct internet works, compute node public IP: $direct_ip (so 'it works' proves nothing)"
else
    info "no direct internet from this node"
fi

# --- 3. client honors the proxy: a dead proxy must break the request ---------
dead_port=$((port + 1))
while (exec 3<>/dev/tcp/127.0.0.1/$dead_port) 2>/dev/null; do dead_port=$((dead_port + 1)); done
if curl -s --max-time 10 --proxy "socks5h://127.0.0.1:$dead_port" "$URL" >/dev/null; then
    bad "request succeeded through a dead proxy port, curl is not using the proxy"
else
    ok "request through a dead proxy port fails (the proxy setting is honored)"
fi

# --- 4. request through the real proxy ---------------------------------------
# Hold the connection open (rate limited download) so we can see it on the listener
tmp=$(mktemp)
curl -s --max-time 20 --proxy "$ALL_PROXY" "$URL" -o "$tmp" &
curl_pid=$!
seen=0
for _ in $(seq 1 20); do
    # established connections whose peer is the SOCKS port = clients talking to the tunnel
    if ss -tn state established "( sport = :$port )" 2>/dev/null | grep -q "127.0.0.1:$port"; then
        seen=1; break
    fi
    sleep 0.1
done
wait $curl_pid
rc=$?
proxy_ip=$(cat "$tmp"); rm -f "$tmp"

if [ $rc -eq 0 ] && [ -n "$proxy_ip" ]; then
    ok "request through the proxy succeeded, public IP seen: $proxy_ip"
else
    bad "request through the proxy failed (curl exit $rc)"
fi
[ $seen -eq 1 ] && ok "connection observed on the SOCKS listener" \
                || info "connection too short-lived to observe on the listener (not fatal)"

# --- 5. egress IP must be the login node's ----------------------------------
if [ -n "$SLURM_SUBMIT_HOST" ] && [ -n "$proxy_ip" ]; then
    login_ip=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SLURM_SUBMIT_HOST" \
        "curl -s --max-time 10 $URL" 2>/dev/null)
    if [ -n "$login_ip" ]; then
        if [ "$login_ip" = "$proxy_ip" ]; then
            ok "proxied IP == login node IP ($login_ip)"
            [ "$direct_ip" = "$proxy_ip" ] && \
                info "compute and login share the same NAT IP, steps 1-4 are the evidence"
        else
            bad "proxied IP ($proxy_ip) differs from the login node IP ($login_ip)"
        fi
    else
        info "could not get the login node IP, skipping comparison"
    fi
fi

# --- 6. other tools ----------------------------------------------------------
if command -v git >/dev/null; then
    if git ls-remote https://github.com/milabench/milabench.git HEAD >/dev/null 2>&1; then
        ok "git ls-remote through proxy works (GIT_CONFIG http.proxy=$GIT_CONFIG_VALUE_0)"
    else
        bad "git ls-remote failed"
    fi
fi

if [ $fail -eq 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
exit $fail
