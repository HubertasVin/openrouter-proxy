"""Simple proxy that reroutes OpenAI-style requests to OpenRouter,
filtering providers by quantization, throughput and privacy policy.

Run: OPENROUTER_API_KEY=sk-or-... PRIVACY_MODE=prioritise_privacy uvicorn main:app --port 8787
"""

import os
import time
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response

OPENROUTER = "https://openrouter.ai/api/v1"
FRONTEND = "https://openrouter.ai/api/frontend/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PRIVACY_MODE = os.environ.get("PRIVACY_MODE", "prioritise_privacy").lower()

app = FastAPI()

# mode -> (allow_prompt_retention, allow_training)
MODE_POLICY = {
    "full_privacy": (False, False),
    "prioritise_privacy": (True, True),
    "can_retain_prompts": (True, False),
    "can_train": (True, True),
}

POLICY_TTL = 3600.0
PICK_TTL = 60.0
_policies: dict[str, dict] = {}
_policies_at = 0.0
_pick_cache: dict[str, tuple[float, list[str], list[dict], dict[str, dict]]] = {}


async def load_policies(client: httpx.AsyncClient) -> dict[str, dict]:
    """provider slug -> dataPolicy (retainsPrompts/training), from OpenRouter's
    frontend endpoint — the /endpoints response carries no privacy fields."""
    global _policies, _policies_at
    if _policies and time.time() - _policies_at < POLICY_TTL:
        return _policies
    try:
        r = await client.get(f"{FRONTEND}/all-providers")
        r.raise_for_status()
        _policies = {p["slug"]: p.get("dataPolicy") or {} for p in r.json().get("data", [])}
        _policies_at = time.time()
    except httpx.HTTPError:
        pass
    return _policies


def provider_slug(ep: dict) -> str:
    return (ep.get("tag") or "").split("/")[0]


def quant_ok(ep: dict) -> bool:
    return ep.get("quantization") in ("int8", "fp8", "mxfp8", "fp4")


def price(ep: dict) -> float:
    pr = ep.get("pricing") or {}
    try:
        return float(pr.get("prompt") or 0) + float(pr.get("completion") or 0)
    except (TypeError, ValueError):
        return float("inf")


def throughput(ep: dict) -> float:
    stats = ep.get("throughput_last_30m") or {}
    try:
        return float(stats.get("p50") or 0)
    except (TypeError, ValueError):
        return 0.0


def is_fully_private(ep: dict, policies: dict[str, dict]) -> bool:
    pol = policies.get(provider_slug(ep))
    if pol is None:
        return False
    return (not pol.get("retainsPrompts", True)
            and not pol.get("training", True)
            and not pol.get("trainingOpenRouter", True))


def privacy_ok(ep: dict, policies: dict[str, dict],
               allow_retention: bool, allow_training: bool) -> bool:
    pol = policies.get(provider_slug(ep))
    if pol is None:
        return allow_retention and allow_training
    return ((allow_retention or not pol.get("retainsPrompts", True))
            and (allow_training or not (pol.get("training", True)
                                        or pol.get("trainingOpenRouter", True))))


def filter_providers(endpoints: list[dict], policies: dict[str, dict],
                     mode: str) -> list[dict]:
    """Single filtering method: quantization + privacy + absolute throughput
    floors. Returns surviving endpoints, cheapest first; the prioritise_privacy
    winner (if any) is moved to the front. The full list is used as
    provider.order so OpenRouter falls back within the filtered set when a
    provider is rate-limited or down."""
    allow_retention, allow_training = MODE_POLICY.get(mode, MODE_POLICY["prioritise_privacy"])
    pool = [e for e in endpoints
            if quant_ok(e) and privacy_ok(e, policies, allow_retention, allow_training)]
    if not pool:
        return []

    survivors = pool
    for floor in (30, 25, 20, 15):
        survivors = [e for e in pool
                     if throughput(e) == 0 or throughput(e) >= floor]
        n_clear = sum(1 for e in survivors if throughput(e) > 0)
        if n_clear >= 2:
            break
    if not any(throughput(e) > 0 for e in survivors):
        # nothing clears even 15 tok/s: keep the fastest rather than none
        top = max(throughput(e) for e in pool)
        if top > 0:
            survivors = [e for e in pool
                         if throughput(e) == 0 or throughput(e) >= top]

    # prices within 1% are tied; prefer the faster endpoint in a tie group
    survivors.sort(key=price)
    tied: list[dict] = []
    ordered: list[dict] = []

    def flush_tied() -> None:
        tied.sort(key=throughput, reverse=True)
        ordered.extend(tied)
        tied.clear()

    for e in survivors:
        if tied and price(e) > price(tied[0]) * 1.01:
            flush_tied()
        tied.append(e)
    flush_tied()
    survivors = ordered

    if mode == "prioritise_privacy":
        cheapest = survivors[0]
        if not is_fully_private(cheapest, policies):
            budget = 2 * price(cheapest)
            for e in survivors:
                if is_fully_private(e, policies) and price(e) <= budget:
                    survivors.remove(e)
                    survivors.insert(0, e)
                    break
    return survivors


async def pick_provider(model: str, auth: str) -> tuple[list[str], list[dict], dict[str, dict]]:
    """Fetch the model's endpoints and pick via filter_providers."""
    key = f"{model}|{PRIVACY_MODE}"
    now = time.time()
    cached = _pick_cache.get(key)
    if cached and now - cached[0] < PICK_TTL:
        return cached[1], cached[2], cached[3]

    base = model.split(":")[0]
    author, _, slug = base.partition("/")
    if not slug:
        return [], [], {}

    async with httpx.AsyncClient(timeout=30) as client:
        policies = await load_policies(client)
        r = await client.get(f"{OPENROUTER}/models/{author}/{slug}/endpoints",
                             headers={"Authorization": auth})
        r.raise_for_status()
        endpoints = (r.json().get("data") or {}).get("endpoints") or []

    picked = filter_providers(endpoints, policies, PRIVACY_MODE)
    tags = [str(e["tag"]) for e in picked if e.get("tag")]
    _pick_cache[key] = (now, tags, endpoints, policies)
    return tags, endpoints, policies


def upstream_headers(request: Request) -> dict[str, str]:
    auth = request.headers.get("Authorization") or (f"Bearer {API_KEY}" if API_KEY else "")
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": request.headers.get("HTTP-Referer", "http://localhost"),
        "X-Title": request.headers.get("X-Title", "openrouter-proxy"),
        "X-OpenRouter-Metadata": "enabled",
    }
    if auth:
        headers["Authorization"] = auth
    return headers


async def relay(request: Request, body: bytes | None):
    headers = upstream_headers(request)
    path = request.url.path.removeprefix("/").removeprefix("v1/").removeprefix("/")
    url = f"{OPENROUTER}/{path}"

    endpoints: list[dict] = []
    policies: dict[str, dict] = {}
    if body is not None:
        try:
            parsed = json.loads(body)
        except ValueError:
            return JSONResponse(status_code=400,
                                content={"error": {"message": "request body is not valid JSON"}})
        model = parsed.get("model", "")
        try:
            tags, endpoints, policies = await pick_provider(model, headers.get("Authorization", ""))
        except httpx.HTTPError:
            tags = []
        print(f"[relay] model={model!r} tags={len(tags)}", flush=True)
        if not tags and PRIVACY_MODE != "can_train":
            return JSONResponse(status_code=502, content={"error": {
                "message": f"no provider satisfies PRIVACY_MODE={PRIVACY_MODE} "
                           f"for '{model or 'unknown model'}'; refusing to route unpinned"}})
        if tags:
            parsed["provider"] = {"order": tags, "allow_fallbacks": False}
        else:
            parsed.pop("provider", None)
        body = json.dumps(parsed).encode()

    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0))
    req = client.build_request(request.method, url, content=body, headers=headers,
                               params=request.url.query)
    resp = await client.send(req, stream=True)

    if resp.status_code >= 400:
        content = await resp.aread()
        await resp.aclose()
        await client.aclose()
        try:
            payload = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            payload = {"error": {"message": content.decode(errors="replace")}}
        return JSONResponse(status_code=resp.status_code, content=payload)

    ctype = resp.headers.get("content-type", "")

    if "text/event-stream" in ctype:
        async def stream():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
        return StreamingResponse(stream(), status_code=resp.status_code,
                                 media_type=ctype)

    content = await resp.aread()
    await resp.aclose()
    await client.aclose()

    return Response(content=content, media_type=ctype or "application/json")


@app.post("/{path:path}")
async def proxy_post(path: str, request: Request):
    return await relay(request, await request.body())


@app.get("/{path:path}")
async def proxy_get(path: str, request: Request):
    return await relay(request, None)
