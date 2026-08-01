/* operate: live meters, sparklines, in-place row updates. Everything on
   this page renders from the tick stream; when the stream drops, every
   live region flips to the stale treatment (dashed border, last-known
   value + timestamp + retry countdown) instead of blanking. */
import { useEffect, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import { Spark, push } from "../lib/spark";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import { fmtClock, fmtGiB, fmtUp } from "../lib/api";
import type { ModelRow, Tick } from "../lib/types";

type Series = Record<string, number[]>;

function useSeries(tick: Tick | null): Series {
  const ref = useRef<Series>({});
  const [, bump] = useState(0);
  useEffect(() => {
    if (!tick) return;
    const s = ref.current;
    for (const g of tick.gpus) {
      s[`gpu:${g.device}`] = push(s[`gpu:${g.device}`] ?? [], g.used_bytes / 2 ** 30);
    }
    s["ram"] = push(s["ram"] ?? [], tick.ram.used_bytes / 2 ** 30);
    for (const m of tick.models) {
      if (m.tok_s != null) s[`tok:${m.name}`] = push(s[`tok:${m.name}`] ?? [], m.tok_s);
      if (m.cache?.hit_ratio != null) {
        s[`hit:${m.name}`] = push(s[`hit:${m.name}`] ?? [], m.cache.hit_ratio * 100);
      }
    }
    bump((n) => n + 1);
  }, [tick]);
  return ref.current;
}

function LiveMark({ stale }: { stale: boolean }) {
  return stale
    ? <span class="live stale"><span class="dot"></span>stale</span>
    : <span class="live"><span class="dot"></span>live</span>;
}

function StaleSub({ lastAt, retryIn }: { lastAt: number | null; retryIn: number | null }) {
  return (
    <div class="sub">
      last known {lastAt != null ? fmtClock(lastAt) : "—"} ·{" "}
      <span>{retryIn != null ? `retrying in ${retryIn}s` : "reconnecting…"}</span>
    </div>
  );
}

/* Mutations POST to the existing console endpoints (they answer with a
   redirect into the old job page; the job itself shows up in the stream,
   so the SPA just toasts and lets the jobs badge/page track it). */
async function submitAction(path: string, label: string) {
  try {
    const r = await fetch(path, { method: "POST", redirect: "manual" });
    if (r.status === 401) {
      location.href = "/login?next=/v2/";
      return;
    }
    // 303 (opaque redirect in manual mode) = the job was submitted.
    if (r.ok || r.type === "opaqueredirect" || r.status === 0 || r.status === 303) {
      toast("ok", `${label} submitted`, "queued as a job · watch it on the jobs page");
    } else {
      toast("err", `✗ ${label} failed`, `server answered ${r.status}`, 9000);
    }
  } catch (e) {
    toast("err", `✗ ${label} failed`, String(e), 9000);
  }
}

function ModelRowView({ m, spark, stale }:
                      { m: ModelRow; spark: number[]; stale: boolean }) {
  const [busy, setBusy] = useState(false);
  const chipClass = m.state_class ? `chip ${m.state_class}` : "chip";
  const act = (verb: "load" | "unload") => {
    setBusy(true);
    submitAction(`/models/${encodeURIComponent(m.name)}/${verb}`,
                 `${verb} ${m.name}`).finally(() => setBusy(false));
  };
  return (
    <tr class="rowlink"
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          location.href = `/profiles/${encodeURIComponent(m.name)}`;
        }}>
      <td>
        {m.name}
        <div class="sub">
          {[m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : "",
            m.file,
            m.backend !== "llama-cpp" ? m.backend : ""]
            .filter(Boolean).join(" · ")}
        </div>
      </td>
      <td class="sub">{m.placement}</td>
      <td><span class={chipClass}><span class="dot"></span>{m.state}</span></td>
      <td class={m.tok_s != null ? "num" : "num sub"}>
        {m.tok_s != null ? m.tok_s.toFixed(1) : "—"}
      </td>
      <td>
        {spark.length > 1
          ? <Spark data={spark} height={26} label={`${m.name} tok/s`} />
          : null}
      </td>
      <td class="actions">
        {m.running
          ? <button disabled={busy || stale} onClick={() => act("unload")}>unload</button>
          : m.registered && m.enabled
            ? <button disabled={busy || stale} onClick={() => act("load")}>load</button>
            : null}
      </td>
    </tr>
  );
}

export function Operate() {
  const { tick, stale, lastAt, retryIn } = useStream();
  const series = useSeries(stale ? null : tick);

  if (!tick) {
    return (
      <div class="widget">
        <span class="sub">
          {stale
            ? <>stream unavailable · {retryIn != null ? `retrying in ${retryIn}s` : "reconnecting…"}</>
            : "connecting to telemetry…"}
        </span>
      </div>
    );
  }

  const { services, gpus, ram, models, jobs } = tick;
  const runningJobs = jobs.filter((j) => j.status === "running").length;
  const cacheModels = models.filter((m) => m.cache != null);
  const wcls = stale ? "widget stale" : "widget";

  return (
    <>
      <div class="widget" style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">
        <span class={services.swap.ok ? "chip ok" : "chip err"}>
          <span class="dot"></span>
          llama-swap · {services.swap.ok ? "running" : "down"} · {services.swap.detail}
          {services.swap.ok && services.swap.latency_ms != null ? ` · ${services.swap.latency_ms}ms` : ""}
        </span>
        <span class={services.api.ok ? "chip ok" : "chip err"}>
          <span class="dot"></span>
          API · {services.api.ok ? "ok" : "down"}
          {services.api.latency_ms != null ? ` · ${services.api.latency_ms}ms` : ""}
        </span>
        {stale && (
          <span class="chip warn"><span class="dot"></span>telemetry · stream dropped</span>
        )}
        <span class="sub num">console up {fmtUp(services.console_started)}</span>
        <span class="grow" style="flex:1"></span>
        <a href="/v2/jobs" class="sub">
          {runningJobs} job{runningJobs === 1 ? "" : "s"} running →
        </a>
      </div>

      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">
        {gpus.map((g) => {
          const used = g.used_bytes / 2 ** 30;
          const total = g.total_bytes / 2 ** 30;
          return (
            <div class={wcls} key={g.device}>
              <div class="label">
                <span>{g.device} · {g.name}</span>
                <LiveMark stale={stale} />
              </div>
              <div class="meterbar">
                <div class="fill" style={`width:${total ? Math.min(100, (used / total) * 100) : 0}%`}></div>
              </div>
              <div class="big">{used.toFixed(1)}<span class="sub"> / {total.toFixed(0)} GiB VRAM</span></div>
              <Spark data={series[`gpu:${g.device}`] ?? []} min={0} max={total}
                     label={`${g.device} VRAM used`} />
              {stale && <StaleSub lastAt={lastAt} retryIn={retryIn} />}
            </div>
          );
        })}

        <div class={wcls}>
          <div class="label">
            <span>system RAM</span>
            <LiveMark stale={stale} />
          </div>
          <div class="meterbar">
            <div class="fill" style={`width:${ram.total_bytes ? Math.min(100, (ram.used_bytes / ram.total_bytes) * 100) : 0}%`}></div>
          </div>
          <div class="big">
            {fmtGiB(ram.used_bytes)}<span class="sub"> / {fmtGiB(ram.total_bytes, 0)} GiB</span>
          </div>
          <Spark data={series["ram"] ?? []} min={0} max={ram.total_bytes / 2 ** 30}
                 label="RAM used" />
          {stale && <StaleSub lastAt={lastAt} retryIn={retryIn} />}
        </div>

        {cacheModels.map((m) => {
          const c = m.cache!;
          const pct = c.hit_ratio != null ? c.hit_ratio * 100 : null;
          return (
            <div class={wcls} key={`cache-${m.name}`}>
              <div class="label">
                <span>
                  {m.name} · MoE cache{" "}
                  <Info label="about the MoE cache">
                    Expert-cache hit ratio for this model, scraped from its
                    runtime metrics. While the cache is still learning which
                    experts recur, the ratio is provisional; "learning done"
                    means placement has settled.
                  </Info>
                </span>
                <LiveMark stale={stale} />
              </div>
              <div class="meterbar">
                <div class="fill" style={`width:${pct ?? 0}%`}></div>
              </div>
              <div class="big">
                {pct != null ? pct.toFixed(1) : "—"}
                <span class="sub"> % hit · learning {c.learning === null ? "?" : c.learning ? "…" : "done"}</span>
              </div>
              <Spark data={series[`hit:${m.name}`] ?? []} min={0} max={100}
                     label={`${m.name} cache hit ratio`} />
              {stale && <StaleSub lastAt={lastAt} retryIn={retryIn} />}
            </div>
          );
        })}
        {cacheModels.length === 0 && (
          <div class="widget">
            <div class="label"><span>MoE cache</span></div>
            <div class="sub" style="margin-top:.4rem">
              no resident model is running with an MoE cache
            </div>
          </div>
        )}
      </div>

      <div class={wcls}>
        <h2>resident models</h2>
        {models.length === 0
          ? <p class="sub">no profiles registered yet — <a href="/add">add a model</a></p>
          : (
            <table>
              <thead>
                <tr>
                  <th>model</th><th>placement</th><th>state</th>
                  <th class="num">tok/s</th><th style="width:130px">last 60 s</th><th></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <ModelRowView key={m.name} m={m}
                                spark={series[`tok:${m.name}`] ?? []} stale={stale} />
                ))}
              </tbody>
            </table>
          )}
        <p class="sub" style="margin:.5rem 0 0">
          rows update in place over SSE · click a row for the model page
        </p>
        {stale && <StaleSub lastAt={lastAt} retryIn={retryIn} />}
      </div>
    </>
  );
}
