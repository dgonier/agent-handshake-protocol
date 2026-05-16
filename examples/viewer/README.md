# AHP debate viewer

Mobile-friendly browser UI for the SLM-driven adversarial debate demo.
Spin it up with Docker Compose; reach it from any device on the same LAN.

## Quick start

```bash
# From the repo root.
docker compose up --build
```

Then browse:

* http://localhost:9876 from the host machine.
* http://<host-lan-ip>:9876 from a phone on the same Wi-Fi.

### LAN access from a phone (WSL2 specifically)

If the host is **Windows + WSL2**, the WSL VM has its own IP that
phones can't reach. Forward Windows host port 9876 to the WSL VM by
running this once in an elevated PowerShell:

```powershell
cd <repo-root>
.\examples\viewer\expose-on-lan.ps1
```

It prints the URL to open on your phone. Tear it back down with
`-Remove`.

On native Linux or macOS hosts none of this is needed — point the
phone straight at the host's LAN IP on port 9876.

The viewer expects AWS credentials reachable through the standard
boto3 chain. The compose file bind-mounts `~/.aws` read-only into
the container — `aws configure` on the host is enough.

## Routes

| Route        | Purpose                                          |
| ------------ | ------------------------------------------------ |
| `GET /`      | Latest debate + form to kick off a new run.      |
| `POST /run`  | Submit a new debate (background task).           |
| `GET /audit` | Audit-event timeline from the latest run.        |
| `GET /runs`  | All persisted runs.                              |
| `GET /runs/{id}` | One specific persisted run.                  |
| `GET /api/latest`| JSON snapshot of the latest run.             |
| `GET /healthz`   | Liveness probe.                              |

## Configuration

Env vars (all optional, sensible defaults baked into the compose file):

| Variable                    | Default                                              |
| --------------------------- | ---------------------------------------------------- |
| `AWS_REGION`                | `us-east-1`                                          |
| `BEDROCK_MODEL_ID`          | `us.anthropic.claude-haiku-4-5-20251001-v1:0`        |
| `VIEWER_CLOUDWATCH_GROUP`   | `/ahp/astrophysics-demo` (empty disables)            |
| `VIEWER_DEFAULT_TOPIC`      | `What caused the Big Bang?`                          |
| `VIEWER_DEFAULT_SUBDOMAIN`  | `astrophysics`                                       |
| `AHP_MODAL_VLLM_URL`        | unset → metadata-only Modal provider stays *off*. Set to the OpenAI-compatible base URL (e.g. a Modal vLLM deployment) to register a second compute provider alongside the self-hosted Bedrock leaf. |
| `AHP_MODAL_VLLM_MODEL`      | `qwen2-5-7b` — slug used in the MenuLeaf address.    |
| `AHP_MODAL_VLLM_TIER`       | `small` — tier mapping for the Modal leaf.           |
| `AHP_MODAL_VLLM_RATE`       | `0.00015` — credits/1k chars charged by Modal.       |
| `AHP_MODAL_VLLM_LATENCY_MS` | `8000` — honest p95 covering vLLM cold starts.       |
| `AHP_MODAL_VLLM_PROVE_HEALTH` | `0` — when `1`, probe `GET {URL}/models` and flip the provider to alive on 2xx. Otherwise it stays metadata-only and isn't routable. |
| `AHP_SECONDARY_ORG`         | `beta` — when the Modal URL is set, a second server is registered under this org with `compute_binding` pointing at the Modal leaf. Set to the same value as the primary org (`tifin` by default) or `AHP_DISABLE_SECONDARY_SERVER=1` to skip. |
| `AHP_SECONDARY_BASE_RATE`   | `0.00025` — the secondary server's posted base rate. |

## Public exposure (optional)

The compose file ships a Cloudflare quick-tunnel sidecar behind a
profile. Bring it up with:

```bash
docker compose --profile tunnel up --build
```

Then watch the `cloudflared` container logs for the public
`https://*.trycloudflare.com` URL. The viewer itself is unauthenticated
— treat the URL as sensitive while the tunnel runs.

Tear down with `Ctrl-C` or `docker compose down`.

## Notes

* Redis is real (compose service), not fakeredis — same protocol code
  runs whether you're on your laptop or a real cluster.
* Transcripts persist to the `viewer_data` named volume; runs survive
  restarts. Remove the volume to clear them.
* Concurrent runs are serialized by a global asyncio lock.
* This example deliberately has no auth and no input validation
  beyond `count ∈ [2,6]`. Don't expose it to the internet long-term
  without putting an auth layer in front.
