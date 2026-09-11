# openrouter-proxy

FastAPI proxy that reroutes OpenAI-style requests to OpenRouter, filtering
providers by quantization, throughput and privacy policy. Point any
OpenAI-compatible client at `http://localhost:8787/v1`.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --port 8787
```

Point any OpenAI-compatible client at `http://localhost:8787/v1` with its own
OpenRouter key — the proxy forwards each client's `Authorization` header
upstream. `OPENROUTER_API_KEY` in the environment is optional: only a fallback
for clients that send no key (e.g. curl testing).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Optional. Fallback key when the client sends no `Authorization`; the client's own key always takes priority. Also used for the endpoint lookup when set (throughput stats require auth). |
| `PRIVACY_MODE` | `prioritise_privacy` | `full_privacy`, `prioritise_privacy`, `can_retain_prompts`, or `can_train`. |
| `PROXY_FOOTER` | `1` | Set `0` to disable the provider-info footer. |

Privacy modes:

- `full_privacy` — only providers that neither retain prompts nor train on them.
- `prioritise_privacy` — cheapest surviving provider; if it retains/trains, a
  fully-private one within 2× its price is preferred.
- `can_retain_prompts` — retention allowed, training not.
- `can_train` — both allowed.

## Install as a service

```bash
curl -fsSL https://raw.githubusercontent.com/HubertasVin/openrouter-proxy/main/install.sh | bash
```

Clones the repo to `~/.local/opt/openrouter-proxy`, creates a Python venv,
installs dependencies, and registers a systemd user service listening on port
8787. No API key is stored — clients (e.g. VS Code BYOK) pass their own key on
every request; set `OPENROUTER_API_KEY` beforehand only if you want a keyless-
client fallback, preserved across reinstalls in `~/.config/openrouter-proxy/env`.

## Filtering

1. Endpoint must be quantized at 8 bits or below (`int8`/`fp8`/`mxfp8`/`fp4`).
2. Privacy must satisfy the selected mode.
3. Absolute throughput floor: starts at 30 tok/s and relaxes through 25, 20
   and 15 tok/s until ≥2 endpoints clear it. Unmeasured endpoints are never
   dropped.
4. Winner = cheapest survivor, with prices within 1% treated as tied and the
   faster endpoint preferred within a tie.

The pick is cached per model+mode for 60 s. A failed lookup routes unpinned in
`can_train` mode; privacy modes return 502 instead of routing unpinned.
