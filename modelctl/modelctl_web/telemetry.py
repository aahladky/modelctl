"""Telemetry for the /v2 console: one SSE stream feeding the operate and
jobs pages -- service status, per-GPU VRAM, system RAM, per-resident-model
tok/s, expert cache hit ratio + learning state, remote fleet node stats, and
the job list.

Every probe is injectable so tests run hermetically (no xpu-smi, no
llama-swap, no /proc reads), and every section of a snapshot degrades
independently: a dead probe yields its section's empty shape, never an
exception out of the stream -- the client renders last-known + stale
instead of a blank page.

tok/s is a real rate: delta of the worker's tokens_predicted_total
counter between ticks, not the server's per-request average gauge. The
gauge is included as tok_s_avg so an idle-but-loaded model still shows
what it last did.

The remote-node block is the one section that cannot be read inline. Its
source is another machine over ssh, so modelctl_nodestats owns a timer
thread and a cache and this module only ever reads that cache -- one
hung ssh must not stall the front page. The thread is not started at all
until the registry actually names a remote node.
"""
import json
import re
import threading
import time
import urllib.request

import modelctl_nodestats

MOE_METRIC_RE = re.compile(r"llamacpp:moe_cache_(\w+)(\{[^}]*\})?\s+(\S+)")
DEVICE_LABEL_RE = re.compile(r'device="([^"]*)"')

# Process-wide shutdown signal for every SSE generator. Open event
# streams otherwise outlive a SIGTERM (uvicorn waits for in-flight
# responses to drain, and a tick loop never drains on its own), so a
# graceful restart of the console hung until systemd's SIGKILL.
# __main__ sets this from uvicorn's exit handler; generators treat it as
# "finish this iteration and end the response".
SHUTDOWN = threading.Event()


def request_shutdown():
    SHUTDOWN.set()


def read_meminfo(path="/proc/meminfo"):
    """{"total_bytes", "available_bytes", "used_bytes"}; zeros when unreadable."""
    total = avail = 0
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {"total_bytes": total, "available_bytes": avail,
            "used_bytes": max(0, total - avail)}


def fetch_metrics_text(port, timeout=2):
    """Raw /metrics exposition text from a worker port, or None."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=timeout) as r:
            return r.read().decode(errors="replace")
    except Exception:
        return None


def parse_worker_metrics(text):
    """One /metrics text -> {"plain": {name: float}, "moe": {device: {k: v}}}.

    Single fetch, two views: the plain name->value map (labels collapsed,
    fine for the unlabeled throughput counters) and the per-device MoE
    cache map (labels matter there -- collapsing them was exactly the bug
    scrape_moe_cache_metrics() exists to avoid).
    """
    plain = {}
    moe = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = MOE_METRIC_RE.match(line)
        if m:
            key, labels, val = m.group(1), m.group(2) or "", m.group(3)
            dm = DEVICE_LABEL_RE.search(labels)
            moe.setdefault(dm.group(1) if dm else "", {})[key] = val
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            plain[name] = float(parts[-1])
        except ValueError:
            continue
    return {"plain": plain, "moe": moe}


def summarize_cache(moe):
    """Per-device expert cache map -> the one-line summary the operate page
    shows: aggregate hit ratio from the raw counters, and "learning" as
    "any device still learning". None when there is nothing to summarize."""
    if not moe:
        return None
    hits = misses = 0
    learning = None
    ratios = []
    for dev, stats in moe.items():
        try:
            hits += int(float(stats.get("hits_total", 0)))
            misses += int(float(stats.get("misses_total", 0)))
        except ValueError:
            pass
        if "learning" in stats:
            try:
                flag = float(stats["learning"]) > 0
            except ValueError:
                flag = None
            if flag is not None:
                learning = flag if learning is None else (learning or flag)
        if "hit_ratio" in stats:
            try:
                ratios.append(float(stats["hit_ratio"]))
            except ValueError:
                pass
    if hits + misses > 0:
        ratio = hits / (hits + misses)
    elif ratios:
        ratio = sum(ratios) / len(ratios)
    else:
        ratio = None
    return {"hit_ratio": ratio, "learning": learning,
            "hits": hits, "misses": misses,
            "devices": sorted(k for k in moe if k)}


def job_row(job, tail_chars=2000):
    """One JobStore row -> the typed shape the SPA consumes."""
    return {
        "id": job["id"],
        "type": job["type"],
        "title": job["title"],
        "lane": job.get("lane") or "mutation",
        "status": job["status"],
        "progress": float(job.get("progress") or 0.0),
        "detail": job.get("detail") or "",
        "error": job.get("error") or "",
        "result_tail": (job.get("result") or "")[-tail_chars:],
        "cancellable": bool(job.get("cancellable", 1)),
        "created": job.get("created"),
        "started": job.get("started"),
        "finished": job.get("finished"),
    }


class TelemetryCollector:
    """Assembles one tick snapshot from injectable probes.

    Defaults talk to the real stack; tests inject fakes. The collector is
    shared across SSE connections (tok/s deltas live in _prev keyed by
    model name), so two open pages don't halve each other's rates --
    per-model state stores the last counter reading with its timestamp
    and every snapshot computes the rate against that.
    """

    def __init__(self, store=None,
                 inventory_fn=None, meminfo_fn=None, runtime_fn=None,
                 profiles_fn=None, metrics_fn=None, swap_probe_fn=None,
                 node_stats_fn=None, node_poller=None):
        self.store = store
        self._inventory_fn = inventory_fn
        self._meminfo_fn = meminfo_fn or read_meminfo
        self._runtime_fn = runtime_fn
        self._profiles_fn = profiles_fn
        self._metrics_fn = metrics_fn or fetch_metrics_text
        self._swap_probe_fn = swap_probe_fn
        self._node_stats_fn = node_stats_fn
        # Constructed here, started nowhere: building the object is free
        # and doing it eagerly means two concurrent SSE connections can
        # never race to create two pollers for one console.
        self._node_poller = node_poller or modelctl_nodestats.NodeStatsPoller()
        self._prev = {}  # model name -> (tokens_predicted_total, monotonic ts)
        self.started = time.time()

    # -- default probes (lazy imports keep module import light) ----------
    def _inventory(self):
        if self._inventory_fn:
            return self._inventory_fn()
        import modelctl
        return modelctl.get_gpu_inventory()

    def _runtime(self):
        if self._runtime_fn:
            return self._runtime_fn()
        from .swap import LlamaSwapClient
        import modelctl
        base = modelctl.LLAMA_SWAP_BASE_URL.rsplit("/v1", 1)[0]
        return LlamaSwapClient(base_url=base, timeout=3).runtime_state()

    def _profiles(self):
        if self._profiles_fn:
            return self._profiles_fn()
        import modelctl
        out = []
        for p in sorted(modelctl.PROFILES_DIR.glob("*.json")):
            try:
                out.append(modelctl.load_profile(p.stem))
            except Exception:
                continue
        return out

    def _swap_probe(self):
        """Service chips: llama-swap health and its API, with latency."""
        if self._swap_probe_fn:
            return self._swap_probe_fn()
        from .swap import LlamaSwapClient, ModelctlSwapError
        import modelctl
        base = modelctl.LLAMA_SWAP_BASE_URL.rsplit("/v1", 1)[0]
        client = LlamaSwapClient(base_url=base, timeout=3)
        port = base.rsplit(":", 1)[-1]
        swap = {"ok": False, "latency_ms": None, "detail": f":{port}"}
        api = {"ok": False, "latency_ms": None, "detail": ""}
        try:
            t0 = time.monotonic()
            client.health()
            swap["ok"] = True
            swap["latency_ms"] = round((time.monotonic() - t0) * 1000)
        except ModelctlSwapError as e:
            swap["detail"] = e.message
        try:
            t0 = time.monotonic()
            client.registered_models()
            api["ok"] = True
            api["latency_ms"] = round((time.monotonic() - t0) * 1000)
        except ModelctlSwapError as e:
            api["detail"] = e.message
        return {"swap": swap, "api": api}

    def _node_stats(self):
        """(block, error) for the remote fleet. Reads a cache, never ssh.

        The poller's thread is started on the first tick that finds a
        remote node in the registry and not before: an install with no
        fleet gets no thread, which is also what makes the test suite
        (whose redirected MODELCTL_HOME has no registry) incapable of
        opening a connection from here.
        """
        if self._node_stats_fn:
            return self._node_stats_fn()
        import modelctl_fleet
        from . import fleet as fleetview
        nodes = list(modelctl_fleet.load_fleet())
        if not nodes:
            # Not a failure and not a card: an install with no registered
            # remote node has nothing to show, which is different from a
            # fleet that exists and cannot be read.
            return modelctl_nodestats.summarize(
                [], {"stats": {}, "age_seconds": None, "ok": False}), None
        self._node_poller.start()
        snap = self._node_poller.snapshot()
        rows = modelctl_nodestats.node_rows(
            nodes, fleetview.presence_state, modelctl_fleet.load_presence(),
            snap.get("stats"))
        block = modelctl_nodestats.summarize(rows, snap)
        return block, (None if block["ok"] else snap.get("error"))

    # -- assembly --------------------------------------------------------
    def _placement(self, profile):
        cfg = profile.get("config", {})
        if cfg.get("split_mode") and cfg.get("tensor_split"):
            base = f"split {cfg['tensor_split']} ({cfg['split_mode']})"
        else:
            base = cfg.get("device") or "(backend default)"
        extras = []
        if "exps=CPU" in (cfg.get("extra") or ""):
            extras.append("CPU experts")
        if "-ot " in (cfg.get("extra") or "") or "--override-tensor" in (cfg.get("extra") or ""):
            extras.append("tensor overrides")
        mc = profile.get("moe_cache") or {}
        if mc.get("mode", "off") != "off":
            extras.append(f"expert cache {mc['mode']}")
        return base + (f" + {', '.join(extras)}" if extras else "")

    def _size_bytes(self, profile):
        path = profile.get("model_path")
        if not path:
            return None
        try:
            import modelctl_vram
            return modelctl_vram.weights_bytes_on_disk(path)
        except Exception:
            return None

    def _placement_summary(self, profile, gpus):
        """The resulting allocation as chips: per-device split share plus
        flags, or None when there is nothing structured to say. Shares
        are percentages of the split weights -- the weights are
        proportions, not GiB, and a summary that claimed bytes it never
        measured would be lying (same rule as the em dash)."""
        cfg = profile.get("config", {})
        devices = []
        flags = []
        split = (cfg.get("tensor_split") or "").strip()
        if split:
            try:
                weights = [float(x) for x in split.split(",") if x.strip()]
            except ValueError:
                weights = []
            total = sum(weights)
            if weights and total > 0:
                names = [g["device"] for g in gpus]
                for i, w in enumerate(weights):
                    name = names[i] if i < len(names) else f"device {i}"
                    devices.append({"name": name,
                                    "text": f"{round(100 * w / total)}%"})
                mode = cfg.get("split_mode") or "layer"
                if mode in ("layer", "row"):
                    flags.append(f"{mode} split")
        elif cfg.get("device"):
            devices.append({"name": cfg["device"], "text": ""})
        extra = cfg.get("extra") or ""
        if "exps=CPU" in extra:
            flags.append("CPU experts")
        if "-ot " in extra or "--override-tensor" in extra:
            flags.append("tensor overrides")
        mc = profile.get("moe_cache") or {}
        if mc.get("mode", "off") != "off":
            flags.append(f"expert cache {mc['mode']}")
        if not devices and not flags:
            return None
        return {"devices": devices, "flags": flags}

    def _model_rows(self, runtime, gpus=None):
        rows = []
        now = time.monotonic()
        seen = set()
        for p in self._profiles():
            name = p.get("name", "")
            seen.add(name)
            rt = runtime.get(name) or {}
            running = bool(rt.get("running"))
            port = rt.get("port")
            tok_s = tok_s_avg = None
            cache = None
            if running and port:
                text = self._metrics_fn(port)
                if text:
                    parsed = parse_worker_metrics(text)
                    plain = parsed["plain"]
                    total = plain.get("llamacpp:tokens_predicted_total")
                    if total is not None:
                        prev = self._prev.get(name)
                        self._prev[name] = (total, now)
                        if prev and now > prev[1] and total >= prev[0]:
                            tok_s = (total - prev[0]) / (now - prev[1])
                    tok_s_avg = plain.get("llamacpp:predicted_tokens_seconds")
                    cache = summarize_cache(parsed["moe"])
            else:
                self._prev.pop(name, None)
            rows.append({
                "name": name,
                "file": p.get("file") or "",
                "backend": p.get("backend") or "llama-cpp",
                "placement": self._placement(p),
                "placement_summary": self._placement_summary(p, gpus or []),
                "state": rt.get("state") or (
                    "stopped" if rt.get("registered") else
                    ("stopped" if rt else "unregistered")),
                "state_class": rt.get("state_class") or "stopped",
                "registered": bool(rt.get("registered")),
                "running": running,
                "enabled": bool(p.get("enabled", True)),
                "port": port,
                "size_bytes": self._size_bytes(p),
                "moe_cache_mode": (p.get("moe_cache") or {}).get("mode", "off"),
                "tok_s": round(tok_s, 2) if tok_s is not None else None,
                "tok_s_avg": tok_s_avg,
                "cache": cache,
            })
        # llama-swap can run models modelctl doesn't manage; show them
        # rather than pretending the runtime is idle.
        for name, rt in runtime.items():
            if name in seen:
                continue
            rows.append({
                "name": name, "file": "", "backend": "unmanaged",
                "placement": "(no modelctl profile)",
                "placement_summary": None,
                "state": rt.get("state") or "unknown",
                "state_class": rt.get("state_class") or "",
                "registered": bool(rt.get("registered")),
                "running": bool(rt.get("running")),
                "enabled": True, "port": rt.get("port"),
                "size_bytes": None, "moe_cache_mode": "off",
                "tok_s": None, "tok_s_avg": None, "cache": None,
            })
        rows.sort(key=lambda r: (not r["running"], r["name"]))
        return rows

    def job_rows(self, limit=50):
        if not self.store:
            return []
        return [job_row(j) for j in self.store.list(limit)]

    def snapshot(self):
        # Every section degrades on its own, and says so. An empty `gpus`
        # or a zeroed `ram` is indistinguishable from the truth of a
        # machine with no GPUs / no memory reading, so a section that
        # threw records why: the page marks that region stale instead of
        # rendering the fallback as fact.
        errors = {}
        try:
            services = self._swap_probe()
        except Exception as e:
            services = {"swap": {"ok": False, "latency_ms": None, "detail": "probe failed"},
                        "api": {"ok": False, "latency_ms": None, "detail": "probe failed"}}
            errors["services"] = str(e) or "probe failed"
        services["console_started"] = self.started
        try:
            gpus = [{"device": g["device"], "name": g["name"],
                     "total_bytes": g["total_bytes"],
                     "free_bytes": g["free_bytes"],
                     "used_bytes": max(0, g["total_bytes"] - g["free_bytes"])}
                    for g in self._inventory()]
        except Exception as e:
            gpus = []
            errors["gpus"] = str(e) or "device inventory failed"
        try:
            ram = self._meminfo_fn()
        except Exception as e:
            ram = {"total_bytes": 0, "available_bytes": 0, "used_bytes": 0}
            errors["ram"] = str(e) or "meminfo read failed"
        try:
            runtime = self._runtime()
        except Exception as e:
            runtime = {}
            errors["runtime"] = str(e) or "runtime probe failed"
        try:
            models = self._model_rows(runtime, gpus)
        except Exception as e:
            models = []
            errors["models"] = str(e) or "profile read failed"
        try:
            jobs = self.job_rows()
        except Exception as e:
            jobs = []
            errors["jobs"] = str(e) or "job store read failed"
        try:
            node_stats, node_error = self._node_stats()
            if node_error:
                errors["node_stats"] = node_error
        except Exception as e:
            node_stats = {"nodes": [], "age_seconds": None, "ok": False,
                          "present": 0, "pins_agree": True, "protocol": ""}
            errors["node_stats"] = str(e) or "node stats unavailable"
        return {"ts": time.time(), "services": services, "gpus": gpus,
                "ram": ram, "models": models, "jobs": jobs,
                "node_stats": node_stats, "errors": errors}


def sse_stream(collector, interval=2.0, max_seconds=3600, stop=None):
    """SSE generator: one `tick` event per interval. Bounded lifetime so a
    forgotten tab reconnects rather than pinning a thread forever (the
    client's EventSource reconnect handles the cut invisibly).

    `stop` (threading.Event, default the module-wide SHUTDOWN) ends the
    stream between ticks; the wait doubles as the tick sleep, so a
    shutdown interrupts the pause instead of riding it out."""
    stop = SHUTDOWN if stop is None else stop
    deadline = time.time() + max_seconds
    try:
        while time.time() < deadline and not stop.is_set():
            snap = collector.snapshot()
            yield f"event: tick\ndata: {json.dumps(snap)}\n\n"
            if stop.wait(interval):
                return
    except (GeneratorExit, BrokenPipeError, ConnectionResetError):
        return
