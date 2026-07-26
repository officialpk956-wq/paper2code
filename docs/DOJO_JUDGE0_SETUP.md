# Dojo Judge0 Execution Engine — Setup

The Dojo grader (`backend/services/dojo_grading.py`) runs untrusted submissions
through a pluggable execution backend selected by the **`DOJO_ENGINE`** env var
(`backend/services/judge0_service.py`):

| `DOJO_ENGINE` | Backend | Use |
|---|---|---|
| `judge0` | self-hosted [Judge0](https://github.com/judge0/judge0) REST API | **production** (needs a dedicated VM) |
| `e2b`    | E2B sandbox (default fallback) | works on Render; per-run sandbox |
| `local`  | plain `subprocess` — **no sandbox** | local dev / CI tests only |

Selection order in `run_batch`: `judge0` (only if `JUDGE0_URL` is set) → `local` → `e2b`.
If `DOJO_ENGINE=judge0` but `JUDGE0_URL` is missing, it logs a warning once and
falls back to `e2b` (grading still works; you just aren't on Judge0).

> The grader sends **one batched program per submission** (all test cases run in a
> single process), so Judge0 receives 1 request per submission, not N.

## Why a dedicated VM (not Render)

Judge0 needs `isolate`, which requires Linux **cgroups v1** and privileged
container capabilities. Render/Heroku-style PaaS don't grant these, so Judge0
must run on a VM you control (a small dedicated box, e.g. a 2 vCPU / 4 GB cloud
VM, is plenty for a coding dojo).

## Provision (Docker Compose)

On an Ubuntu VM with Docker + Compose:

```bash
# 1. cgroups v1 (Judge0/isolate requirement) — add to kernel cmdline, then reboot
sudo sed -i 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 systemd.unified_cgroup_hierarchy=0"/' /etc/default/grub
sudo update-grub && sudo reboot

# 2. pull the published compose stack
wget https://github.com/judge0/judge0/releases/download/v1.13.1/judge0-v1.13.1.zip
unzip judge0-v1.13.1.zip && cd judge0-v1.13.1
```

Edit `judge0.conf`:
- Set `REDIS_PASSWORD` and `POSTGRES_PASSWORD` to strong values.
- Set **`AUTHN_HEADER=X-Auth-Token`** and **`AUTHN_TOKEN=<a long random secret>`** so the API requires a token (do NOT run it open to the internet).
- Optionally restrict `ALLOW_ORIGIN` and bind to a private network / behind a firewall.

```bash
docker compose up -d db redis           # start datastores first
sleep 10
docker compose up -d                    # then the server + workers
```

Verify:

```bash
curl -s -H "X-Auth-Token: <token>" http://<vm-ip>:2358/about
# submit a quick sanity job:
curl -s -H "X-Auth-Token: <token>" -H "Content-Type: application/json" \
  "http://<vm-ip>:2358/submissions?base64_encoded=false&wait=true" \
  -d '{"language_id":71,"source_code":"print(2+2)","stdin":""}'
# -> expect {"stdout":"4\n", "status":{"id":3,"description":"Accepted"}, ...}
```

`language_id` **71** is Python 3 in the stock Judge0 image (`GET /languages` to confirm).

## Wire it into the backend

Set these on the FastAPI service (Render dashboard → Environment):

| Env var | Value |
|---|---|
| `DOJO_ENGINE` | `judge0` |
| `JUDGE0_URL` | `http://<vm-ip>:2358` (prefer HTTPS via a reverse proxy) |
| `JUDGE0_AUTH_TOKEN` | the `AUTHN_TOKEN` you set above |
| `JUDGE0_PYTHON_LANG_ID` | `71` (only if your image differs) |

Redeploy. Submissions now execute on Judge0. Nothing else in the app changes —
the grader's contract (`run_batch`) is identical across engines.

## Hardening checklist

- Firewall: only the backend's egress IP may reach `:2358`; never expose it publicly.
- Keep `AUTHN_TOKEN` set — an open Judge0 is a public RCE service.
- Cap `MAX_CPU_TIME_LIMIT`, `MAX_MEMORY_LIMIT`, `MAX_PROCESSES_AND_OR_THREADS` in `judge0.conf`.
- Put the VM on a private network / VPC peering with the backend if your host supports it.
- Monitor the worker queue depth; scale `count` of workers in compose for load.
