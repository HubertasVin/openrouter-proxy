"""Simple proxy that reroutes OpenAI-style requests to OpenRouter,
filtering providers by quantization, throughput and privacy policy.

Run: OPENROUTER_API_KEY=sk-or-... PRIVACY_MODE=prioritise_privacy uvicorn main:app --port 8787
"""

import os
import time
import json
import hashlib
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response

OPENROUTER = "https://openrouter.ai/api/v1"
FRONTEND = "https://openrouter.ai/api/frontend/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PRIVACY_MODE = os.environ.get("PRIVACY_MODE", "prioritise_privacy").lower()
SHOW_FOOTER = os.environ.get("PROXY_FOOTER", "1").lower() not in ("0", "false", "no")

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
RUN_TTL = float(os.environ.get("RUN_TTL", 3600.0))  # agent-run accumulator expiry
_policies: dict[str, dict] = {}
_policies_at = 0.0
_pick_cache: dict[str, tuple[float, list[str], list[dict], dict[str, dict]]] = {}
# run-key -> {"calls", "tokens", "cost", "ms", "t"} for the current agent run
# (one user prompt -> Copilot's tool-loop makes several LLM calls)
_runs: dict[str, dict] = {}


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


def is_8bit(ep: dict) -> bool:
    return ep.get("quantization") in ("int8", "fp8", "mxfp8")


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
    """Single filtering method: 8-bit quantization + privacy + adaptive
    throughput floor. Returns surviving endpoints, cheapest first; the
    prioritise_privacy winner (if any) is moved to the front. The full list
    is used as provider.order so OpenRouter falls back within the filtered
    set when a provider is rate-limited or down."""
    allow_retention, allow_training = MODE_POLICY.get(mode, MODE_POLICY["prioritise_privacy"])
    pool = [e for e in endpoints
            if is_8bit(e) and privacy_ok(e, policies, allow_retention, allow_training)]
    if not pool:
        return []

    measured = [e for e in pool if throughput(e) > 0]
    survivors = pool
    if measured:
        fastest = max(throughput(e) for e in measured)
        for factor in (0.4, 0.3, 0.2):
            floor = factor * fastest
            survivors = [e for e in pool
                         if throughput(e) == 0 or throughput(e) >= floor]
            n_clear = sum(1 for e in survivors if throughput(e) > 0)
            if n_clear - 1 >= 2 or factor == 0.2:
                break

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
    """Fetch the model's endpoints and pick via filter_providers.
    Returns (winning endpoint tags in fallback order, full endpoint list,
    policy map) — the latter two feed the response footer's stats."""
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


def privacy_label(ep: dict, policies: dict[str, dict]) -> str:
    pol = policies.get(provider_slug(ep))
    if pol is None:
        return "policy unknown"
    retains = pol.get("retainsPrompts", True)
    trains = pol.get("training", True) or pol.get("trainingOpenRouter", True)
    if not retains and not trains:
        return "fully private"
    if retains and trains:
        return "retains + trains"
    return "retains prompts" if retains else "trains on prompts"


def fmt_throughput(ep: dict) -> str | None:
    t = throughput(ep)
    return f"{t:g} tok/s" if t > 0 else None


def fmt_latency(ep: dict) -> str | None:
    lat = ep.get("latency_last_30m")
    if isinstance(lat, dict):
        lat = lat.get("p50")
    if isinstance(lat, (int, float)) and lat > 0:
        return f"{lat:g} ms p50"
    return None


def fmt_ms(ms: float) -> str:
    return f"{ms:g} ms" if ms < 1000 else f"{ms / 1000:g} s"


def fmt_price(ep: dict) -> str | None:
    pr = ep.get("pricing") or {}
    try:
        p = float(pr.get("prompt") or 0) * 1e6
        c = float(pr.get("completion") or 0) * 1e6
    except (TypeError, ValueError):
        return None
    return f"${round(p, 4):g}/${round(c, 4):g} per M"


def first_user_text(messages: list) -> str:
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def run_key(model: str, messages: list) -> str | None:
    """Identity of one agent run (one user prompt): Copilot's tool loop repeats
    the same first user message across its follow-up calls."""
    text = first_user_text(messages)
    if not text:
        return None
    return hashlib.sha256(f"{model}|{text}".encode()).hexdigest()[:16]


def update_run(key: str | None, usage: dict | None, total_ms: float) -> dict | None:
    """Fold this call's metrics into the run accumulator; return the cumulative
    run stats (calls/tokens/cost/ms) to render in the footer."""
    if not key:
        return None
    now = time.time()
    for k in [k for k, v in _runs.items() if now - v["t"] > RUN_TTL]:
        del _runs[k]
    run = _runs.setdefault(key, {"calls": 0, "tokens": 0, "cost": 0.0, "ms": 0.0, "t": now})
    run["calls"] += 1
    run["ms"] += total_ms
    run["t"] = now
    if isinstance(usage, dict):
        n = usage.get("completion_tokens")
        if isinstance(n, (int, float)) and n > 0:
            run["tokens"] += int(n)
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost > 0:
            run["cost"] += cost
    return run


def fmt_run(run: dict | None) -> str | None:
    if not run or run["calls"] < 2:
        return None  # single-call run: the per-call footer already says it all
    bits = [f"{run['calls']} calls"]
    if run["tokens"]:
        bits.append(f"{run['tokens']:,} tok")
    if run["cost"]:
        bits.append(f"${round(run['cost'], 5):g}")
    if run["ms"]:
        bits.append(f"{fmt_ms(run['ms'])} total")
    return f"*Run: {' · '.join(bits)}*"


def fmt_measured_tps(usage: dict | None, measured: dict | None) -> str | None:
    """Request-measured tok/s: completion tokens over the relayed stream time.
    Shown when the endpoints lookup lacks stats (unauthenticated lookup)."""
    if not isinstance(usage, dict) or not isinstance(measured, dict):
        return None
    total_ms = measured.get("total_ms")
    n = usage.get("completion_tokens")
    if not isinstance(total_ms, (int, float)) or total_ms <= 0:
        return None
    if not isinstance(n, (int, float)) or n <= 0:
        return None
    return f"{n / (total_ms / 1000):.1f} tok/s"


def fmt_cost(usage: dict | None) -> str | None:
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)):
        return None
    return f"${round(cost, 5):g}"


def match_endpoint(meta: dict, endpoints: list[dict]) -> tuple[dict | None, dict | None]:
    """Find the metadata's selected endpoint and the matching stats entry
    (matched by provider_name, disambiguated by the served model slug)."""
    eps = (meta.get("endpoints") or {}).get("available") or []
    sel = next((e for e in eps if e.get("selected")), None)
    if not sel:
        return None, None
    provider = sel.get("provider") or ""
    served = sel.get("model") or ""
    cands = [e for e in endpoints if e.get("provider_name") == provider]
    if len(cands) > 1 and served:
        named = [e for e in cands if served in (e.get("name") or "")]
        if named:
            cands = named
    return sel, (cands[0] if cands else None)


def provider_footer(meta: dict, endpoints: list[dict], policies: dict[str, dict],
                    usage: dict | None = None,
                    measured: dict | None = None,
                    run: dict | None = None) -> str | None:
    """Markdown footer describing the endpoint that served the request, with
    stats from the endpoints lookup (throughput/latency/price/privacy) plus
    this request's own measured ttft/total when available, plus cumulative
    stats for the agent run (Copilot tool-loop calls sharing one prompt)."""
    sel, ep = match_endpoint(meta, endpoints)
    if not sel:
        return None
    bits = [f"**{sel.get('provider') or '?'}**"]
    if ep:
        q = ep.get("quantization")
        if q and q != "unknown":
            bits.append(q)
        for piece in (fmt_throughput(ep), fmt_latency(ep), fmt_price(ep),
                      privacy_label(ep, policies)):
            if piece:
                bits.append(piece)
        if ep.get("supports_implicit_caching"):
            bits.append("implicit cache")
    cost = fmt_cost(usage)
    mtps = fmt_measured_tps(usage, measured)
    if mtps:
        bits.append(mtps)
    if measured:
        v = measured.get("total_ms")
        if isinstance(v, (int, float)) and v > 0:
            bits.append(f"total {fmt_ms(v)}")
    footer = f"\n\n---\n*Routed via {' · '.join(bits)}*\n*Total cost: {cost if cost else ''}*"
    run_line = fmt_run(run)
    if run_line:
        footer += f"\n{run_line}"
    return footer


def sse_delta(obj: dict, footer: str) -> bytes:
    chunk = {
        "id": obj.get("id"),
        "object": "chat.completion.chunk",
        "created": obj.get("created"),
        "model": obj.get("model"),
        "choices": [{"index": 0, "delta": {"content": footer}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def sse_rewrite(line: bytes, footer: str) -> bytes:
    """Append footer text to a data line's delta.content."""
    obj = json.loads(line[6:])
    ch = (obj.get("choices") or [{}])[0]
    delta = ch.setdefault("delta", {})
    delta["content"] = (delta.get("content") or "") + footer
    return f"data: {json.dumps(obj)}".encode()


async def sse_with_footer(chunks, endpoints: list[dict], policies: dict[str, dict],
                          t0: float | None = None, rkey: str | None = None):
    """Pass upstream SSE through with the provider-info footer folded into the
    last content chunk. A footer delta emitted after the finish_reason chunk is
    dropped by clients that finalize the message there (VS Code chat), so the
    footer must ride on content no later than the stop chunk. The last content
    line, the stop line and everything after are buffered until [DONE]; the
    footer is then folded into the buffered content line (or the stop line when
    the response had no content), which is emitted before the stop line."""
    buf = b""
    usage: dict | None = None
    meta_obj: dict | None = None
    held: bytes | None = None
    stop_line: bytes | None = None
    tail: list[bytes] = []
    injected = False
    ttft_ms: float | None = None

    async for chunk in chunks:
        if t0 is not None and ttft_ms is None:
            ttft_ms = (time.monotonic() - t0) * 1000  # first upstream byte
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.rstrip(b"\r")

            if text == b"data: [DONE]":
                if SHOW_FOOTER and meta_obj is not None:
                    total_ms = (time.monotonic() - t0) * 1000 if t0 is not None else 0.0
                    measured = {"total_ms": total_ms} if t0 is not None else {}
                    run = update_run(rkey, usage, total_ms)
                    footer = provider_footer(meta_obj["openrouter_metadata"],
                                             endpoints, policies, usage, measured, run)
                    if footer:
                        injected = True
                        target = held if held is not None else stop_line
                        if target is not None:
                            target = sse_rewrite(target, footer)
                            if held is not None:
                                held = target
                            else:
                                stop_line = target
                for out in (held, stop_line):
                    if out is not None:
                        yield out + b"\n"
                held = stop_line = None
                for t in tail:
                    yield t + b"\n"
                tail = []
                yield text + b"\n"
                continue

            obj = None
            if text.startswith(b"data: "):
                try:
                    obj = json.loads(text[6:])
                except ValueError:
                    obj = None
            if isinstance(obj, dict) and obj.get("usage"):
                usage = obj["usage"]
            if isinstance(obj, dict) and obj.get("openrouter_metadata") and meta_obj is None:
                meta_obj = obj

            is_content = (isinstance(obj, dict) and obj.get("choices")
                          and isinstance((obj["choices"][0].get("delta") or {}).get("content"), str)
                          and obj["choices"][0]["delta"]["content"] != "")
            has_finish = isinstance(obj, dict) and obj.get("choices") and obj["choices"][0].get("finish_reason")

            if is_content:
                if held is not None:
                    yield held + b"\n"
                held = text
            elif has_finish and stop_line is None:
                stop_line = text
            else:
                if stop_line is not None:
                    tail.append(text)
                else:
                    yield text + b"\n"

    for out in (held, stop_line):
        if out is not None:
            yield out + b"\n"
    for t in tail:
        yield t + b"\n"
    if buf:
        yield buf
    print(f"[footer] stream: metadata={'yes' if meta_obj is not None else 'no'} "
          f"injected={injected} ttft={ttft_ms and round(ttft_ms)}ms", flush=True)


async def relay(request: Request, body: bytes | None):
    headers = upstream_headers(request)
    path = request.url.path.removeprefix("/").removeprefix("v1/").removeprefix("/")
    url = f"{OPENROUTER}/{path}"

    endpoints: list[dict] = []
    policies: dict[str, dict] = {}
    rkey: str | None = None
    if body is not None:
        try:
            parsed = json.loads(body)
        except ValueError:
            return JSONResponse(status_code=400,
                                content={"error": {"message": "request body is not valid JSON"}})
        model = parsed.get("model", "")
        rkey = run_key(model, parsed.get("messages"))
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
        if "messages" in parsed and "usage" not in parsed:
            parsed["usage"] = {"include": True}
        body = json.dumps(parsed).encode()

    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0))
    req = client.build_request(request.method, url, content=body, headers=headers,
                               params=request.url.query)
    t0 = time.monotonic()
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
                async for chunk in sse_with_footer(resp.aiter_bytes(), endpoints,
                                                   policies, t0, rkey):
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
        return StreamingResponse(stream(), status_code=resp.status_code,
                                 media_type=ctype)

    content = await resp.aread()
    total_ms = (time.monotonic() - t0) * 1000
    await resp.aclose()
    await client.aclose()

    if SHOW_FOOTER and "application/json" in ctype:
        try:
            payload = json.loads(content)
            run = update_run(rkey, payload.get("usage"), total_ms)
            footer = provider_footer(payload.get("openrouter_metadata") or {},
                                     endpoints, policies, payload.get("usage"),
                                     {"total_ms": total_ms}, run)
            print(f"[footer] json: metadata={'yes' if payload.get('openrouter_metadata') else 'no'} "
                  f"footer={'yes' if footer else 'no'}", flush=True)
            if footer and payload.get("choices"):
                msg = payload["choices"][0].get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] += footer
                    content = json.dumps(payload).encode()
        except (ValueError, KeyError, IndexError):
            pass

    return Response(content=content, media_type=ctype or "application/json")


@app.post("/{path:path}")
async def proxy_post(path: str, request: Request):
    return await relay(request, await request.body())


@app.get("/{path:path}")
async def proxy_get(path: str, request: Request):
    return await relay(request, None)
