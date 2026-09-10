# OpenRouter provider-filtering proxy

Single-file FastAPI proxy. Copilot sends OpenAI-style requests here; the proxy
fetches the model's endpoint list from OpenRouter, filters it, and pins the
request to the winning endpoint via `provider.order`.

## Run

```bash
pip install -r requirements.txt
OPENROUTER_API_KEY=sk-or-... PRIVACY_MODE=prioritise_privacy uvicorn main:app --port 8787
```

Point Copilot's OpenAI-compatible endpoint at `http://localhost:8787/v1`.

## Environment variables

- `OPENROUTER_API_KEY` — fallback key if the client doesn't send one. Also used
  for the endpoint lookup: `throughput_last_30m` is only returned to
  authenticated requests, so without a key the throughput floor is skipped.
- `PRIVACY_MODE` — one of:
  - `full_privacy` — only providers that neither retain prompts nor train on them.
  - `prioritise_privacy` — pick the cheapest surviving provider; if it retains/trains,
    pick a fully-private one within 2× its price, else the cheapest.
  - `can_retain_prompts` — prompt retention allowed, training not.
  - `can_train` — both allowed.
- `PROXY_FOOTER` — set to `0` to disable the provider-info footer that is
  appended to response content (on by default).

## Provider-info footer

Because VS Code chat renders only message content, the proxy appends a markdown
footer describing how the request was routed:

> *🔀 routed via **Morph** · served `z-ai/glm-5.3-flash-20260826` · fp8 ·
> 120.5 tok/s · 310 ms · $0.15/$0.6 per M tok · fully private · implicit cache ·
> $0.00213 · iad · attempt 2*

- The actually-serving endpoint comes from OpenRouter's router metadata
  (opt-in header `X-OpenRouter-Metadata: enabled`); on streaming responses the
  footer is injected as an extra content delta just before `data: [DONE]`.
- Throughput (p50 tok/s), latency (p50 ms), list price per million tokens,
  quantization, and privacy label come from the endpoints lookup and the cached
  policy map; fields with no data are omitted.
- The request's actual cost comes from `usage.cost`, enabled by sending
  `usage: {"include": true}` upstream (added automatically to chat requests).
- There is no per-request cache-hit-rate metric; the footer instead notes
  `implicit cache` when the serving endpoint supports implicit caching.

## Data sources

- Endpoints: `GET /api/v1/models/{author}/{slug}/endpoints` — quantization,
  pricing, `throughput_last_30m` percentiles (tokens/sec, p50 used).
- Privacy: `GET /api/frontend/v1/all-providers` — per-provider `dataPolicy`
  (`retainsPrompts`, `training`, `trainingOpenRouter`), cached 1 h. The
  endpoints response carries no privacy fields. Providers missing from the
  policy map are treated as worst-case (retains + trains) and only admitted in
  `can_train` mode.

## Filtering (all in `filter_providers`)

1. Endpoint must be 8-bit quantized (`int8`/`fp8`/`mxfp8`).
2. Privacy must satisfy the selected mode.
3. Throughput floor starts at 40% of the fastest endpoint's p50 and relaxes to
   30%, then 20%, until at least 2 endpoints besides the fastest clear it.
   Skipped entirely when no throughput data is visible. Endpoints without
   throughput data are never dropped by the floor (they can't be evaluated) —
   only measured endpoints are gated.
4. Winner = cheapest survivor (with the privacy-priority override above). In
   `prioritise_privacy`, the 2× budget is relative to the cheapest survivor:
   if that one is free, only free fully-private endpoints qualify.

The pick is cached per model+mode for 60 s. A failed lookup routes unpinned in
`can_train` mode; in the privacy modes it returns **502** instead — unpinned
routing could land on a provider that retains or trains on prompts, silently
violating the mode. Malformed request JSON returns 400. GET requests (e.g.
`/v1/models`) pass through untouched.
