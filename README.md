# openrouter-proxy

FastAPI proxy that reroutes OpenAI-style requests to OpenRouter, filtering
providers by quantization, throughput and privacy policy. Point any
OpenAI-compatible client at `http://localhost:8787/v1`.

## Run

```bash
pip install -r requirements.txt
OPENROUTER_API_KEY=sk-or-... uvicorn main:app --port 8787
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Fallback key when the client sends no `Authorization`. Also used for the endpoint lookup (throughput stats require auth). |
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
8787. Set `OPENROUTER_API_KEY` before running, or edit
`~/.config/openrouter-proxy/env` afterwards.

## Filtering

1. Endpoint must be 8-bit quantized (`int8`/`fp8`/`mxfp8`).
2. Privacy must satisfy the selected mode.
3. Adaptive throughput floor: starts at 40% of the fastest endpoint's p50
   tok/s, relaxes to 30% then 20% until ≥2 other endpoints clear it. Skipped
   when no throughput data is visible; unmeasured endpoints are never dropped.
4. Winner = cheapest survivor, with prices within 1% treated as tied and the
   faster endpoint preferred within a tie.

The pick is cached per model+mode for 60 s. A failed lookup routes unpinned in
`can_train` mode; privacy modes return 502 instead of routing unpinned.
